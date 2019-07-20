#include <iostream>
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <utility>

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

// Erik
constexpr double STRENGTH = 2980.56;

constexpr int NUM_SKILLS = 3;
constexpr bool reduce_states = true;

constexpr int SKILL_SP[3] = {2868, 5883, 4711};
constexpr int SKILL_STARTUP = 6;
constexpr int SKILL_RECOVERY[3] = {111, 114, 54};
constexpr int SKILL_MOD[3] = {1036, 933, 0};

// UI gets hidden this long when you skill; you can't
// queue another skill until you wait for the UI to come back
constexpr int UI_RECOVERY = 114;

// Axe
constexpr int COMBO_SP[5] = {200, 240, 360, 380, 420};
constexpr int COMBO_STARTUP = 16;
constexpr int COMBO_RECOVERY[5] = {46, 61, 40, 78, 19};
constexpr int COMBO_MOD[5] = {114, 122, 204, 216, 228};

constexpr int FS_SP = 300;
constexpr int FS_MOD = 192;
constexpr int FS_STARTUP = 40 + 78;
constexpr int FS_RECOVERY = 34;

constexpr int XFS_STARTUP[5] = {68, 62, 65, 67, 40};

// Not the combo counter, but your location in a standard C5 combo.  We
// use this to figure out what startup/recovery frames are relevant.
//
// Combo values:
// -1 0 1 2 3 4       99 (FS)
//   x x x x x x
//     \-------/
//     \--------------/
//
// NB: -1 means you didn't come out of a combo; 4 means
// you just finished a c5.  You cycle then to state 0.
// State -1 happens when you skill.
//
// 99 means you did an FS.  Importantly, 99 % 5 == 4,
// so this also wraps around to zero so we can conveniently
// implement next_combo_state.
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
      /*
    default:
      return static_cast<ComboState>((combo + 1) % 5);
      */
  }
}

// This is UB but I don't care
using AdvStateCode = uint64_t;
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

static_assert(sizeof(AdvState) == sizeof(AdvStateCode), "AdvState packs");

std::ostream& operator<<(std::ostream& os, AdvState st) {
  os << "[sp=" << st.s.sp_[0] << "," << st.s.sp_[1] << ";c=" << (int)st.s.combo_ << ";b=" << st.s.buff_frames_left_ << "]";
  return os;
}

constexpr AdvState INIT_ST = { .s =
  { .sp_ = {0, 0, 0}, .combo_ = NO_COMBO, .buff_frames_left_ = 0 } };

// 'f' = force strike
// 'x' = basic combo
// '1' = S1
// '2' = S2
using ActionCode = char;

// Map from AdvState k to AdvStates which, when taking ActionCode, result
// in k
using ComesFrom = std::unordered_map<AdvStateCode, std::vector<std::pair<AdvState, ActionCode>>>;

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

using NId = int;
using PId = int;

struct P { std::unordered_set<NId> nids_; };
struct N { PId pid_; AdvStateCode st_; };

int main(int argc, char** argv) {

  int frames;
  std::cerr << "frames? ";
  std::cin >> frames;
  std::cerr << frames << "\n";

  auto states = compute_states();

  std::cerr << "num_states = " << states.size() << "\n";

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
    // dumb
    for (const auto& kv : action_inverse_set) {
      action_inverse[kv.first] = std::vector<std::pair<AdvState, char>>(kv.second.begin(), kv.second.end());
    }
    initial_state = ns[c2n[INIT_ST.c]].pid_;
  } else {
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

  std::cerr << "final num states = " << num_states << "\n";

  // Compute necessary frame window
  int max_frames = 1;
  for (int s = 0; s < num_states; s++) {
    for (auto pair : action_inverse[s]) {
      AdvState p_st;
      ActionCode ac;
      std::tie(p_st, ac) = pair;
      int frames = compute_frames(p_st, ac, AdvState{ .c = ix2state[s] });
      if (frames > max_frames) {
        max_frames = frames + 1;
      }
    }
  }

  std::cerr << "max frame window = " << max_frames << "\n";

  std::vector<double> best_dps(max_frames * num_states, -1);
  std::vector<std::string> best_sequence(max_frames * num_states);

  auto dix = [&](int frame, int state_ix) {
    return (frame % max_frames) * num_states + state_ix;
  };

  best_dps[dix(0, initial_state)] = 0;

  double last_best = 0;
  // This is the bottleneck!
  //  - Snapshotting (so I can continue computing later)
  //  - Double => Float
  //  - More state reduction?
  //    - Unsound approximations
  //  - Branch bound (we KNOW that this is provably worse,
  //    prune it)
  //    - Same combo, same buff, dps is less, SP is less
  //    - Best case "catch up" for states
  //  - Improve locality of access?
  //    - Only five actions: bucket them together
  //    - Lay out action_inverses contiguously, so we don't
  //      thrash cache
  //  - Estimate time left
  //  - Implement best sequence in a less shitty way
  //    - Bit pack
  //    - "Everything is roughly the same size, do the
  //      allocation inline"
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
    for (int s = 0; s < num_states; s++) {
      AdvState st;
      st.c = ix2state[s];
      auto& cur = best_dps[dix(f, s)];
      auto& cur_seq = best_sequence[dix(f, s)];

      // Consider all states which could have lead here
      for (auto pair : action_inverse[s]) {
        AdvState p_st;
        ActionCode ac;
        std::tie(p_st, ac) = pair;

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
            if (tmp >= 0 && tmp > cur) {
              // std::cerr << "f" << f << " " << ac << " ps" << p_st << " s" << st << " accepting " << tmp << " (" << dmg << ")\n";
              cur = tmp;
              cur_seq = best_sequence[z] + ac;
            }
          }
        }
      }
    }
    double best = -1;
    std::string best_seq = "";
    for (int s = 0; s < num_states; s++) {
      auto tmp = best_dps[dix(f, s)];
      if (tmp > best) {
        best = tmp;
        best_seq = best_sequence[dix(f, s)];
      }
    }
    if (best >= 0) {
      if (best >= 0 && best > last_best) {
        if (1) {
          int combo_count = 0;
          auto print_combo = [&](bool trailing_space = true) {
            if (combo_count) {
              std::cerr << "c" << combo_count;
              if (trailing_space) {
                std::cerr << " ";
              }
              combo_count = 0;
            }
          };
          for (auto c : best_seq) {
            switch (c) {
              case 'x':
                if (combo_count == 5) {
                  print_combo();
                }
                combo_count++;
                break;
              case 'f':
                print_combo(/*trailing_space*/ false);
                std::cerr << "fs ";
                break;
              case '1':
                print_combo();
                std::cerr << "s1   ";
                break;
              case '2':
                print_combo();
                std::cerr << "s2   ";
                break;
              case '3':
                print_combo();
                std::cerr << "s3   ";
                break;
              default:
                assert(0);
            }
          }
          print_combo();
        } else {
          std::cerr << best_seq << " ";
        }
        std::cerr << "=> " << best << " dmg in " << f << " frames\n";
        last_best = best;
      }
    }
  }

  return 0;
}
