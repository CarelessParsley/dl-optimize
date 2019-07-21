#include <iostream>
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <utility>
#include <cassert>
#include <chrono>

namespace std {
  template<typename T>
  inline void hash_combine(std::size_t& seed, const T& val) {
    std::hash<T> hasher;
    seed ^= hasher(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  // taken from https://stackoverflow.com/a/7222201/916549
  template<typename S, typename T>
  struct hash<std::pair<S, T>> {
    inline size_t operator()(const std::pair<S, T>& val) const {
      size_t seed = 0;
      hash_combine(seed, val.first);
      hash_combine(seed, val.second);
      return seed;
    }
  };
}

// Absolute tolerance when comparing DPS floating point for equality.
constexpr double EPSILON = 0.01;

// Configuration
constexpr int NUM_SKILLS = 2; // How many skills to consider calculation for
constexpr bool reduce_states = true; // Whether or not to apply state space reduction with Hopcroft

// Erik's stats
constexpr double STRENGTH = 2980.56;

constexpr int SKILL_SP[3] = {2868, 5883, 4711};
constexpr int SKILL_STARTUP = 6;
constexpr int SKILL_RECOVERY[3] = {111, 114, 54};
constexpr int SKILL_MOD[3] = {1036, 933, 0};

// Axe's stats
constexpr int COMBO_SP[5] = {200, 240, 360, 380, 420};
constexpr int COMBO_STARTUP = 16;
constexpr int COMBO_RECOVERY[5] = {46, 61, 40, 78, 19};
constexpr int COMBO_MOD[5] = {114, 122, 204, 216, 228};

constexpr int FS_SP = 300;
constexpr int FS_MOD = 192;
constexpr int FS_STARTUP = 40 + 78;
constexpr int FS_RECOVERY = 34;

constexpr int XFS_STARTUP[5] = {68, 62, 65, 67, 40};

// UI gets hidden this long when you skill; you can't
// queue another skill until you wait for the UI to come back
constexpr int UI_RECOVERY = 114;

// Not the combo counter, but your location in a standard C5 combo.  We
// use this to figure out what startup/recovery frames are relevant.
// Here's the state machine (implemented by next_combo_basic), where x
// is a basic.
//
// AFTER_FS  --x--V
// AFTER_S$N --x--V
// NO_COMBO -x-> AFTER_C1 -x-> ... -x-> AFTER_C5
//                  ^-----------x-----------/
//
// I used to play the trick where all of the leading to C1 states
// were 4 mod 5, but there's not enough space in an 8-bit integer
// to represent all of them this way.
enum ComboState : int8_t {
  AFTER_S1 = -4,
  AFTER_S2 = -3,
  AFTER_S3 = -2,
  NO_COMBO = -1,
  AFTER_C1 = 0,
  AFTER_C2 = 1,
  AFTER_C3 = 2,
  AFTER_C4 = 3,
  AFTER_C5 = 4,
  AFTER_FS = 5,
};

// After doing another basic, what is your new combo state?
ComboState next_combo_state(ComboState combo) {
  switch (combo) {
    case AFTER_S1:
    case AFTER_S2:
    case AFTER_S3:
    case AFTER_FS:
    case NO_COMBO:
      return AFTER_C1;
    case AFTER_C1:
      return AFTER_C2;
    case AFTER_C2:
      return AFTER_C3;
    case AFTER_C3:
      return AFTER_C4;
    case AFTER_C4:
      return AFTER_C5;
    case AFTER_C5:
      return AFTER_C1;
    default:
      assert(false);
  }
}

// TODO: Do this a more standard complaint way (lol)
using AdvStateCode = uint64_t;

// Description of the "state" of an adventurer.  Currently
// hardcoded to support a single buff.
union AdvState {
  struct {
    int16_t sp_[3];
    int8_t combo_ : 4;
    int16_t buff_frames_left_ : 12;
  } s;
  AdvStateCode c;
};

bool operator==(AdvState a, AdvState b) {
  return a.c == b.c;
}

namespace std {
  template <>
  struct hash<AdvState> {
    size_t operator()(AdvState st) const
    {
      return std::hash<AdvStateCode>()(st.c);
    }
  };
}

static_assert(sizeof(AdvState) == sizeof(AdvStateCode), "AdvState packs into a 64-bit integer");

std::ostream& operator<<(std::ostream& os, AdvState st) {
  os << "[sp=" << st.s.sp_[0] << "," << st.s.sp_[1] << ";c=" << (int)st.s.combo_ << ";b=" << st.s.buff_frames_left_ << "]";
  return os;
}

// Initial state at the start of the sim.
constexpr AdvState INIT_ST = { .s =
  { .sp_ = {0, 0, 0}, .combo_ = NO_COMBO, .buff_frames_left_ = 0 } };

// A single action you can take as the character
// 'f' = force strike
// 'x' = basic combo
// '1' = S1
// '2' = S2
// '3' - S3
using ActionCode = char;

// Design notes:
//
// Originally, I stored individual actions (e.g., ActionCode) in a
// string.  In this
// setting, on S1/S2 only Erik, combo length is something like
// frames/53+1 (this underestimates is some cases).  After 60s you end
// up with rotation length 67.
//
// However, action code currently has 5 elements (x f s1 s2 s3),
// which means you can't actually fit it in two bits.  So instead we
// opted for a variable length code that uses 4-bits, but can encode
// runs of basic combo up to five.
//
// Empirically, encoding 60 frames of actions requires only 25
// in the variable length coding.  This is how we chose 16 bytes
// (32 codes).  This also satisfies our memory budget: we use
// 9G to compute Erik with S3.

// Order of this class matters!  It determines what the "preferred"
// action at any given point in time is (bottom is "most preferred")
// So for example suppose we have c1fs c5fs and c5fs c1fs as possible
// optimal combos; we prefer to take the latter ("front loading" long
// combos).
enum class ActionFragment : uint8_t { // actually only four bit
  NIL = 0,
  FS,
  // NB: We rely on C1-C5/C1FS-C5FS being interleaved in this way.
  // You can reorder them but then you need to edit ActionString::pack()
  // to handle it correctly.  I think C5 should be preferred over C4FS
  // which is why I interleaved, but that's something that's up to
  // taste.
  C1,
  C1FS,
  C2,
  C2FS,
  C3,
  C3FS,
  C4,
  C4FS,
  C5,
  C5FS,
  S1,
  S2,
  S3,
  // 15 (one free slot)
};

// Fixed size encoding of action strings.  Supports "compressed" action
// string of size up to 32, in 16 bytes of space.  NIL terminated.
struct ActionString {
  std::array<uint8_t, 16> buffer_ = {};
  static ActionFragment i2f(uint8_t c) {
    return static_cast<ActionFragment>(c);
  }
  static uint8_t f2i(ActionFragment c) {
    return static_cast<uint8_t>(c);
  }
  static ActionFragment _unpack(uint8_t c, int i) {
    if (i == 0) {
      return i2f((c & 0xF0) >> 4);
    } else {
      return i2f(c & 0x0F);
    }
  }
  static uint8_t _pack(ActionFragment first, ActionFragment second) {
    // NB: It matters that the first element is higher order bits;
    // this means that comparison treats first as most significant,
    // which coincides with lexicographic ordering
    return (f2i(first) << 4) | f2i(second);
  }
  static int _null_at(uint8_t c) {
    if (c == 0) return 0;
    if (_unpack(c, 1) == ActionFragment::NIL) {
      assert(_unpack(c, 0) != ActionFragment::NIL);
      return 1;
    }
    return -1;
  }
  // Get the ith action fragment in an action string.  Valid to read
  // beyond the end of the action string, but not beyond the internal
  // buffer.  In practice this means we can only actually store
  // strings of size 31 (since we don't have any out-of-bounds
  // checking).
  ActionFragment get(int i) const {
    uint8_t c = buffer_[i / 2];
    return _unpack(c, i % 2);
  }
  // Assignment to af[i] not supported.  vector<bool>, rest in peace!
  ActionFragment operator[](int i) const {
    return get(i);
  }
  // Set the ith action fragment in an action string.
  void set(int i, ActionFragment f) {
    assert(i < 32);
    uint8_t c = buffer_[i / 2];
    ActionFragment first = _unpack(c, 0);
    ActionFragment second = _unpack(c, 1);
    if (i % 2 == 0) {
      first = f;
    } else {
      second = f;
    }
    buffer_[i / 2] = _pack(first, second);
  }
  // Push an action code to an action string.  The code will
  // be coalesced with the latest action fragment if possible.
  void push(ActionCode ac) {
    int loc = -1;
    for (int i = 0; i < 16; i++) {
      int j = _null_at(buffer_[i]);
      if (j != -1) {
        loc = i * 2 + j;
        break;
      }
    }
    assert(loc != -1);
    // Check if we can absorb this into the
    // previous entry
    if (loc != 0) {
      ActionFragment p_c = get(loc - 1);
      switch (ac) {
        case 'x':
          switch (p_c) {
            case ActionFragment::C1:
            case ActionFragment::C2:
            case ActionFragment::C3:
            case ActionFragment::C4:
              set(loc - 1, i2f(f2i(p_c) + 2));
              return;
          }
        case 'f':
          switch (p_c) {
            case ActionFragment::C1:
            case ActionFragment::C2:
            case ActionFragment::C3:
            case ActionFragment::C4:
            case ActionFragment::C5:
              set(loc - 1, i2f(f2i(p_c) + 1));
              return;
          }
      }
    }
    ActionFragment c;
    switch (ac) {
      case 'x':
        c = ActionFragment::C1;
        break;
      case 'f':
        c = ActionFragment::FS;
        break;
      case '1':
        c = ActionFragment::S1;
        break;
      case '2':
        c = ActionFragment::S2;
        break;
      case '3':
        c = ActionFragment::S3;
        break;
      default:
        assert(0);
    }
    set(loc, c);
  }
};

std::ostream& operator<<(std::ostream& os, const ActionString& as) {
  for (int i = 0; i < 32; i++) {
    ActionFragment f = as.get(i);
    switch(f) {
      case ActionFragment::NIL:
        return os;
      case ActionFragment::C1:
        os << "c1 ";
        break;
      case ActionFragment::C2:
        os << "c2 ";
        break;
      case ActionFragment::C3:
        os << "c3 ";
        break;
      case ActionFragment::C4:
        os << "c4 ";
        break;
      case ActionFragment::C5:
        os << "c5 ";
        break;
      case ActionFragment::C1FS:
        os << "c1fs ";
        break;
      case ActionFragment::C2FS:
        os << "c2fs ";
        break;
      case ActionFragment::C3FS:
        os << "c3fs ";
        break;
      case ActionFragment::C4FS:
        os << "c4fs ";
        break;
      case ActionFragment::C5FS:
        os << "c5fs ";
        break;
      case ActionFragment::FS:
        os << "fs ";
        break;
      case ActionFragment::S1:
        os << "s1  ";
        break;
      case ActionFragment::S2:
        os << "s2  ";
        break;
      case ActionFragment::S3:
        os << "s3  ";
        break;
      default:
        assert(0);
    }
  }
  return os;
}


// Map from AdvState k to AdvStates which, when taking ActionCode, result
// in k
using ComesFrom = std::unordered_map<AdvStateCode, std::vector<std::pair<AdvState, ActionCode>>>;

// Compute the number of frames it takes to recover from the previous
// action (as per ComboState).  This doesn't account for actions which
// cancel the recovery.
int _handle_recovery(ComboState c, bool combo_recovery) {
  switch (c) {
    case AFTER_FS:
      return FS_RECOVERY;
    case AFTER_S1:
      return SKILL_RECOVERY[0];
    case AFTER_S2:
      return SKILL_RECOVERY[1];
    case AFTER_S3:
      return SKILL_RECOVERY[2];
    case NO_COMBO:
      return 0;
    default:
      if (combo_recovery) {
        return COMBO_RECOVERY[c];
      }
      return 0;
  }
}

// Compute the number of frames it takes to get from p_st to st,
// performing action ac.  Recovery frames after ac are not included
// (they will be accounted for when we process actions from st.)
int compute_frames(AdvState p_st, ActionCode ac, AdvState st) {
  int frames = 0;
  if (ac == 'x') {
    frames += _handle_recovery(static_cast<ComboState>(p_st.s.combo_), /*combo_recovery*/ true);
    if (st.s.combo_ == AFTER_C1) {
      frames += COMBO_STARTUP;
    }
  } else if (ac == 'f') {
    // Force strike cancels combo recovery; instead,
    // there's specific startup costs in this case
    frames += _handle_recovery(static_cast<ComboState>(p_st.s.combo_), /*combo recovery*/ false);
    switch (p_st.s.combo_) {
      case NO_COMBO:
      case AFTER_S1:
      case AFTER_S2:
      case AFTER_S3:
      case AFTER_FS:
        frames += FS_STARTUP;
        break;
      default:
        frames += XFS_STARTUP[p_st.s.combo_];
        break;
    }
  } else {
    // Skill cancels all non-skill recovery; and skill recovery
    // may be increased by UI being hidden (human only; AI not
    // affected by this.)
    switch (p_st.s.combo_) {
      case AFTER_S1:
        frames += std::max(SKILL_RECOVERY[0], UI_RECOVERY);
        break;
      case AFTER_S2:
        frames += std::max(SKILL_RECOVERY[1], UI_RECOVERY);
        break;
      case AFTER_S3:
        frames += std::max(SKILL_RECOVERY[2], UI_RECOVERY);
        break;
    }
    frames += SKILL_STARTUP;
  }
  return frames;
}

// Compute the entire set of reachable AdvStates from the initial sim
// state.  Returns a map from reachable AdvState, to the (state, action)
// pairs which could lead to it (the inverse of the transition
// function.)
ComesFrom compute_states() {
  ComesFrom comes_from;
  std::vector<AdvState> todo{INIT_ST};
  comes_from[INIT_ST.c];
  while (todo.size()) {
    AdvState s = todo.back();
    todo.pop_back();
    auto push = [&](AdvState n_s, ActionCode ac) {
      if (comes_from.count(n_s.c) == 0) {
        todo.emplace_back(n_s);
      }
      comes_from[n_s.c].emplace_back(s, ac);
    };
    int skill_count = 0;
    for (int i = 0; i < NUM_SKILLS; i++) {
      if (s.s.sp_[i] >= SKILL_SP[i]) {
        AdvState t = s;
        t.s.sp_[i] = 0;
        t.s.combo_ = static_cast<ComboState>(AFTER_S1 + i);
        // S3 buff
        if (i == 2) {
          t.s.buff_frames_left_ = 20 * 60;
          //t.s.buff_frames_left_ = 0;
        } else {
          t.s.buff_frames_left_ = std::max(0, t.s.buff_frames_left_ - compute_frames(s, '1' + i, t));
        }
        push(t, '1' + i);
        skill_count++;
      }
    }
    if (skill_count != NUM_SKILLS) {
      // basic
      {
        AdvState t = s;
        t.s.combo_ = next_combo_state(static_cast<ComboState>(t.s.combo_));
        for (int i = 0; i < NUM_SKILLS; i++) {
          t.s.sp_[i] = std::min(t.s.sp_[i] + COMBO_SP[t.s.combo_], SKILL_SP[i]);
        }
        t.s.buff_frames_left_ = std::max(0, t.s.buff_frames_left_ - compute_frames(s, 'x', t));
        push(t, 'x');
      }
      // force strike
      {
        AdvState t = s;
        t.s.combo_ = AFTER_FS;
        for (int i = 0; i < NUM_SKILLS; i++) {
          t.s.sp_[i] = std::min(t.s.sp_[i] + FS_SP, SKILL_SP[i]);
        }
        t.s.buff_frames_left_ = std::max(0, t.s.buff_frames_left_ - compute_frames(s, 'f', t));
        push(t, 'f');
      }
    }
  }
  return comes_from;
}

using AdvStateId = int;
using PartitionId = int;

// Auxiliary definitions for Hopcroft

using NId = int;
using PId = int;

struct P { std::unordered_set<NId> nids_; };
struct N { PId pid_; AdvStateCode st_; };

int main(int argc, char** argv) {

  // Make IO faster
  std::ios_base::sync_with_stdio(false);

  int frames;
  std::cerr << "frames? ";
  std::cin >> frames;
  std::cerr << frames << "\n";

  auto states = compute_states();

  std::cerr << "num states = " << states.size() << "\n";

  std::vector<P> ps; // partitions
  std::vector<N> ns;
  std::unordered_map<AdvStateCode, NId> c2n;

  if (reduce_states) {
    // We need to setup some auxiliary structs.  First,
    // we need to construct an initial partition of adventurer
    // states.  This is done by mapping every adventurer
    // state to a "coarse" state, a canonical state that
    // buckets together every state that could co-exist.

    auto coarsen = [](AdvStateCode st) {
      AdvState c_st;
      c_st.c = st;
      c_st.s.sp_[0] = 0;
      c_st.s.sp_[1] = 0;
      c_st.s.sp_[2] = 0;
      c_st.s.buff_frames_left_ = c_st.s.buff_frames_left_ > 0 ? 1 : 0;
      return c_st.c;
    };

    // This mapping gives us our initial partitioning.

    {
      std::unordered_map<AdvStateCode, PId> coarse2p;
      for (const auto& kv : states) {
        AdvStateCode c = kv.first;
        AdvStateCode coarse = coarsen(c);
        NId n = ns.size();
        ns.emplace_back();
        auto it = coarse2p.find(coarse);
        PId p;
        if (it == coarse2p.end()) {
          p = ps.size();
          ps.emplace_back();
          coarse2p.insert({coarse, p});
        } else {
          p = it->second;
        }
        ns[n].pid_ = p;
        ns[n].st_ = c;
        ps[p].nids_.insert(n);
        c2n[c] = n;
        AdvState st;
        st.c = c;
      }
    }

    std::cerr << "minimum states = " << ps.size() << "\n";

    for (PId p = 0; p < ps.size(); p++) {
      std::cerr << p << " has " << ps[p].nids_.size() << " states like " << AdvState{.c = coarsen(ns[*ps[p].nids_.begin()].st_)} << "\n";
    }

    std::unordered_set<std::pair<PId, ActionCode>> waiting;
    for (PId p = 0; p < ps.size(); p++) {
      for (auto ac : {'x', 'f', '1', '2', '3'}) {
        waiting.insert({p, ac});
      }
    }

    // Do Hopcroft's algorithm (in a shitty inefficient way)

    // while WAITING not empty do
    while (waiting.size()) {
      // select and delete any integer i from WAITING
      PId p;
      ActionCode ac;
      std::tie(p, ac) = *waiting.begin();
      waiting.erase({p, ac});

      // INVERSE <- f^-1(B[i])
      std::unordered_set<NId> inverse;
      for (NId n : ps[p].nids_) {
        for (const auto& kv : states[ns[n].st_]) {
          if (kv.second != ac) continue;
          inverse.insert(c2n[kv.first.c]);
        }
      }

      // for each j such that B[j] /\ INVERSE != {} and
      // B[j] not subset of INVERSE (this list of j is
      // stored in jlist)
      std::unordered_map<PId, std::vector<NId>> jlist;
      for (NId n : inverse) {
        PId q = ns[n].pid_;
        jlist[q].emplace_back(n);
        if (jlist[q].size() == ps[q].nids_.size()) {
          jlist.erase(q);
        }
      }
      for (auto q_qns : jlist) {
        // q <- q+ 1
        // create a new block B[q]
        PId r = ps.size();
        ps.emplace_back();
        PId q = q_qns.first;
        const auto& qns = q_qns.second;
        // B[q] <- B[j] /\ INVERSE
        ps[r].nids_.insert(qns.begin(), qns.end());
        // B[j] <- B[j] - B[q]
        for (NId n : qns) {
          ns[n].pid_ = r;
          ps[q].nids_.erase(n);
        }
        // if j is in WAITING, then add q to WAITING
        for (auto ac : {'x', 'f', '1', '2', '3'}) {
          if (waiting.count({q, ac})) {
            waiting.insert({r, ac});
          } else {
            // if |B[j]| <= |B[q]| then
            //  add j to WAITING
            // else add q to WAITING
            if (ps[r].nids_.size() <= ps[q].nids_.size()) {
              waiting.insert({r, ac});
            } else {
              waiting.insert({q, ac});
            }
          }
        }
      }
    }
    std::cerr << "reduced states = " << ps.size() << "\n";
  }

  // We want to make a table of the states, so assign them a contiguous
  // numbering
  std::unordered_map<AdvStateCode, AdvStateId> state2ix;
  std::vector<AdvStateCode> ix2state;

  int num_states;
  ComesFrom action_inverse;

  // More efficient memory usage by avoiding padding.
  std::vector<AdvState> packed_inverse_state;
  std::vector<ActionCode> packed_inverse_code;
  std::vector<int> inverse_index;  // map of AdvStateId to where it lives in the pack

  PId initial_state;
  if (reduce_states) {
    ix2state.resize(ps.size());
    for (NId n = 0; n < ns.size(); n++) {
      state2ix[ns[n].st_] = ns[n].pid_;
      ix2state[ns[n].pid_] = ns[n].st_; // overwritten
    }
    num_states = ix2state.size();
    std::unordered_map<PId, std::unordered_set<std::pair<AdvState, char>>> action_inverse_set;
    for (const auto& s : states) {
      for (auto kv : s.second) {
        AdvState c;
        char ac;
        std::tie(c, ac) = kv;
        PId p = ns[c2n[c.c]].pid_;
        action_inverse_set[ns[c2n[s.first]].pid_].insert({AdvState{.c = ix2state[p]}, ac});
      }
    }
    // Compute size of pack
    int packed_size = 0;
    for (const auto& kv : action_inverse_set) {
      packed_size += kv.second.size();
    }

    packed_inverse_state.reserve(packed_size);
    packed_inverse_code.reserve(packed_size);
    inverse_index.reserve(ps.size() + 1); // so we can "off by one"

    int packed_i = 0;
    for (int p = 0; p < ps.size(); p++) {
      inverse_index.emplace_back(packed_i);
      for (const auto& st_ac : action_inverse_set[p]) {
        packed_inverse_state.emplace_back(st_ac.first);
        packed_inverse_code.emplace_back(st_ac.second);
        packed_i++;
      }
    }
    assert(packed_i == packed_size);
    inverse_index.emplace_back(packed_i);

    initial_state = ns[c2n[INIT_ST.c]].pid_;
  } else {
    // TODO: fix me
    ix2state.reserve(states.size());
    for (auto s : states) {
      PId p = ix2state.size();
      state2ix.insert({s.first, ix2state.size()});
      ix2state.push_back(s.first);
      action_inverse[p] = s.second;
    }
    num_states = states.size();
    initial_state = state2ix[INIT_ST.c];
  }

  ps.clear();
  ns.clear();

  action_inverse.clear();

  // Compute necessary frame window
  int max_frames = 1;
  for (int p = 0; p < num_states; p++) {
    for (int j = inverse_index[p]; j < inverse_index[p+1]; j++) {
      AdvState p_st = packed_inverse_state[j];
      ActionCode ac = packed_inverse_code[j];
      int frames = compute_frames(p_st, ac, AdvState{ .c = ix2state[p] });
      if (frames > max_frames) {
        max_frames = frames + 1;
      }
    }
  }

  std::cerr << "max frame window = " << max_frames << "\n";

  int buffer_size = max_frames * num_states;
  std::cerr << "projected memory usage = " << (buffer_size * (sizeof(float) + sizeof(ActionString))) / (1 << 20) << " MB\n";

  std::vector<float> best_dps(buffer_size, -1);
  std::vector<ActionString> best_sequence(max_frames * num_states);

  auto dix = [&](int frame, int state_ix) {
    return (frame % max_frames) * num_states + state_ix;
  };

  best_dps[dix(0, initial_state)] = 0;

  auto start_time = std::chrono::high_resolution_clock::now();
  auto last_print_time = start_time;

  float last_best = 0;
  // Correctness improvements
  //   - We currently pick an arbitrary combo among all transpositions
  //     but it would be better if we did some sort of "greedy" pick
  //     where we preferentially kept combo sequences which were
  //     closer to the greedy rotation  Or at the very least, pick
  //     some deterministic combo sequence.
  //     - Can we assign a number and use this to pick winners?
  //       But how to keep the number up to date?
  //     - Dumb solution: do a lexicographic compare on strings
  //       (ew, that'll make it slower)


  // This is the bottleneck!
  //  - Snapshotting (so I can continue computing later)
  //  - Reduce memory usage
  //    - [DONE] Double => Float
  //    - Compress to 16-bit integer?  Would go up to 65535 damage;
  //      for S1/S2 only this is just slightly over.  Quantize further?
  //      For example, use worse prints.
  //      - Many factors in the damage calculation don't matter, except
  //        for determining rounding; e.g.,
  //        strength and initial 5/3.  Could apply mod, with skill
  //        and crit modifiers.
  //            Compare 653 dmg from c1
  //            Without other bits: 114 * (1 + 0.18 * 0.22)
  //        Rounding errors could be severe, however!  (Maybe can
  //        randomize which way we round to to avoid problem?)
  //      - This does't help too much: we spend most of our space
  //        storing combo strings
  //    - [DONE] Previously action strings were dynamically allocated
  //      but now they are all allocated ahead of time, so you'll OOM
  //      immediately when we make the table
  //  - More state reduction?
  //    - Unsound approximations; e.g., quantize buff / SP time
  //  - Branch bound (we KNOW that this is provably worse,
  //    prune it)
  //    - Same combo, same buff, dps is less, SP is less
  //    - Best case "catch up" for states
  //    - Problem: How to know you've been dominated?  Not so easy
  //      to tell without more scanning.
  //  - [TODO] Improve locality of access?
  //    - Only five actions: bucket them together
  //    - Lay out action_inverses contiguously, so we don't
  //      thrash cache
  //  - Estimate time left
  //  - Take advantage of sparsity
  //    - Early on, most states are not accessible.  Only
  //      at a certain load factor should we densify.
  //  - Parallelize/Vectorize...
  //    - ...computation of all incoming actions (no
  //      data dependency, reduction at the end)
  //    - ...computation of all states at the same
  //      frame (no data dependency, reduction at the end)
  //    - ...all frames within the minimum frame window
  //      (provably no data dependency.)
  //  - Small optimizations
  //    - Compute best as we go (in the main loop), rather
  //      than another single loop at the end
  //
  //  - Get some C++ library and build with it
  for (int f = 1; f < frames; f++) {
    auto cur_time = std::chrono::high_resolution_clock::now();
    if (cur_time > last_print_time + 1 * std::chrono::seconds(1)) {
      std::cerr << "fps: " << (f * std::chrono::seconds(1)) / (cur_time - start_time) << "\n";
      last_print_time = cur_time;
    }
    for (int s = 0; s < num_states; s++) {
      AdvState st;
      st.c = ix2state[s];
      auto& cur = best_dps[dix(f, s)];
      auto& cur_seq = best_sequence[dix(f, s)];

      // Consider all states which could have lead here
      for (int j = inverse_index[s]; j < inverse_index[s+1]; j++) {
        AdvState p_st = packed_inverse_state[j];
        ActionCode ac = packed_inverse_code[j];

        int frames = compute_frames(p_st, ac, st);

        int mod = 0;
        if (ac == 'x') {
          mod = COMBO_MOD[st.s.combo_];
        } else if (ac == 'f') {
          mod = FS_MOD;
        } else {
          mod = SKILL_MOD[ac - '1'];
        }

        if (f >= frames) {
          auto z = dix(f - frames, state2ix.at(p_st.c));
          if (best_dps[z] >= 0) {

            double dmg = 5./3;
            dmg *= STRENGTH;
            // dmg *= ability;
            // dmg *= buffs;
            // dmg *= coab;
            dmg *= ((double)mod)/100;
            if (ac == '1' || ac == '2' || ac == '3') {
              dmg *= 1.4; // skill_ab, from KFM + FitF
              // dmg *= skill_buffs;
              // dmg *= skill_coab;
            } else if (ac == 'f') {
              dmg *= 1.3; // Erik's ability
            }
            // dmg *= punisher;
            // dmg *= elemental;
            dmg /= 10. * 1.;  // defense * defense change
            dmg *= (1. + (0.04 + 0.14 /*KFM*/) * (0.7 + 0.15 /* FitF */ + (p_st.s.buff_frames_left_ > 0 ? 0.50 : 0))); // crit rate * crit damage

            auto tmp = best_dps[z] + dmg;
            if (tmp >= 0 && tmp > cur + EPSILON) {
              // std::cerr << "f" << f << " " << ac << " ps" << p_st << " s" << st << " accepting " << tmp << " (" << dmg << ")\n";
              cur = tmp;
              cur_seq = best_sequence[z];
              cur_seq.push(ac);
            } else if (tmp >= 0 && tmp > cur - EPSILON) {
              ActionString tmp_seq = best_sequence[z];
              tmp_seq.push(ac);
              if (std::lexicographical_compare(
                    cur_seq.buffer_.begin(), cur_seq.buffer_.end(),
                    tmp_seq.buffer_.begin(), tmp_seq.buffer_.end())) {
                cur = tmp;
                cur_seq = std::move(tmp_seq);
              }
            }
          }
        }
      }
    }
    float best = -1;
    int best_index = -1;
    int density = 0;
    for (int s = 0; s < num_states; s++) {
      auto tmp = best_dps[dix(f, s)];
      if (tmp > best + EPSILON) {
        best = tmp;
        best_index = dix(f, s);
      }
      if (tmp > 0) {
        density++;
      }
    }
    if (best >= 0) {
      if (best >= 0 && best > last_best + EPSILON) {
        int combo_count = 0;
        std::cout << best_sequence[best_index];
        std::cout << "=> " << best << " dmg in " << f << " frames\n";
        last_best = best;
      }
    }
  }

  return 0;
}
