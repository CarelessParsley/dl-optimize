#include <iostream>
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>

// Erik
constexpr int SKILL_SP[2] = {2868, 5883};
constexpr int SKILL_STARTUP = 6;
constexpr int SKILL_FRAMES[2] = {111, 114};
constexpr int SKILL_DMG[2] = {1036, 933};

// Axe
constexpr int COMBO_SP[5] = {200, 240, 360, 380, 420};
constexpr int COMBO_STARTUP = 16;
constexpr int COMBO_RECOVERY[5] = {46, 61, 40, 78, 19};
constexpr int COMBO_DMG[5] = {114, 122, 204, 216, 228};

constexpr int FS_SP = 300;
constexpr int FS_DMG = 192;
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
  NO_COMBO = -1,
  AFTER_C1 = 0,
  AFTER_C2 = 1,
  AFTER_C3 = 2,
  AFTER_C4 = 3,
  AFTER_C5 = 4,
  AFTER_FS = 99,
};

ComboState next_combo_state(ComboState combo) {
  return ComboState((combo + 1) % 5);
}

// This is UB but I don't care
using AdvStateCode = uint64_t;
union AdvState {
  struct {
    int16_t sp_[2];
    ComboState combo_;
  } s;
  AdvStateCode c;
};
static_assert(sizeof(AdvState) == sizeof(AdvStateCode), "AdvState packs");

constexpr AdvState INIT_ST = { .s = { .sp_ = {0, 0}, .combo_ = NO_COMBO } };

// 'f' = force strike
// 'x' = basic combo
// '1' = S1
// '2' = S2
using ActionCode = char;

// Map from AdvState k to AdvStates which, when taking ActionCode, result
// in k
using ComesFrom = std::unordered_map<AdvStateCode, std::vector<std::pair<AdvState, ActionCode>>>;

ComesFrom compute_states() {
  ComesFrom comes_from;
  std::vector<AdvState> todo{INIT_ST};
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
    for (int i = 0; i < 2; i++) {
      if (s.s.sp_[i] >= SKILL_SP[i]) {
        AdvState t = s;
        t.s.sp_[i] = 0;
        t.s.combo_ = NO_COMBO;
        push(t, '1' + i);
        skill_count++;
      }
    }
    if (skill_count != 2) {
      // basic
      {
        AdvState t = s;
        t.s.combo_ = next_combo_state(t.s.combo_);
        for (int i = 0; i < 2; i++) {
          t.s.sp_[i] = std::min(t.s.sp_[i] + COMBO_SP[t.s.combo_], SKILL_SP[i]);
        }
        push(t, 'x');
      }
      // force strike
      {
        AdvState t = s;
        t.s.combo_ = AFTER_FS;
        for (int i = 0; i < 2; i++) {
          t.s.sp_[i] = std::min(t.s.sp_[i] + FS_SP, SKILL_SP[i]);
        }
        push(t, 'f');
      }
    }
  }
  return comes_from;
}

using AdvStateId = int;

int main() {
  auto states = compute_states();

  // We want to make a table of the states, so assign them a contiguous
  // numbering
  std::unordered_map<AdvStateCode, AdvStateId> state2ix;
  std::vector<AdvStateCode> ix2state;
  ix2state.reserve(states.size());
  for (auto s : states) {
    state2ix.insert({s.first, ix2state.size()});
    ix2state.push_back(s.first);
  }
  auto num_states = states.size();

  std::cerr << "num_states = " << num_states << "\n";

  int frames = 3600; // 60s
  //frames = 600; // 1s
  std::vector<int> best_dps(frames * num_states, -99999999);
  std::vector<std::string> best_sequence(frames * num_states);

  auto dix = [&](int frame, int state_ix) {
    return frame * num_states + state_ix;
  };

  // Initialize frame 0
  //for (int s = 0; s < num_states; s++) {
  //  best_dps[dix(0, s)] = -99999999;
  //}
  best_dps[dix(0, state2ix[INIT_ST.c])] = 0;

  int last_best = 0;
  for (int f = 1; f < frames; f++) {
    for (int s = 0; s < num_states; s++) {
      AdvState st;
      st.c = ix2state[s];
      auto& cur = best_dps[dix(f, s)];
      auto& cur_seq = best_sequence[dix(f, s)];

      // Consider all states which could have lead here
      for (auto pair : states[st.c]) {
        AdvState p_st;
        ActionCode ac;
        std::tie(p_st, ac) = pair;

        int frames = 0;
        int dmg = 0;
        if (ac == 'x') {
          if (p_st.s.combo_ == NO_COMBO) {
            frames += COMBO_STARTUP;
          } else if (p_st.s.combo_ == AFTER_FS) {
            frames += FS_RECOVERY + COMBO_STARTUP;
          } else {
            frames += COMBO_RECOVERY[p_st.s.combo_];
            if (p_st.s.combo_ == AFTER_C5) {
              frames += COMBO_STARTUP;
            }
          }
          dmg = COMBO_DMG[st.s.combo_];
        } else if (ac == 'f') {
          if (p_st.s.combo_ == NO_COMBO) {
            frames += FS_STARTUP;
          } else if (p_st.s.combo_ == AFTER_FS) {
            frames += FS_STARTUP + FS_RECOVERY;
          } else {
            frames += XFS_STARTUP[p_st.s.combo_];
          }
          dmg = FS_DMG;
        } else {
          frames += SKILL_STARTUP;
          frames += SKILL_FRAMES[ac - '1'];
          dmg = SKILL_DMG[ac - '1'];
        }

        if (f >= frames) {
          auto z = dix(f - frames, state2ix[p_st.c]);
          auto tmp = best_dps[z] + dmg;
          if (tmp > cur) {
            cur = tmp;
            cur_seq = best_sequence[z] + ac;
          }
        }
      }
    }
    int best = -1;
    std::string best_seq = "";
    for (int s = 0; s < num_states; s++) {
      auto tmp = best_dps[dix(f, s)];
      if (tmp > best) {
        best = tmp;
        best_seq = best_sequence[dix(f, s)];
      }
    }
    if (best >= 0 && best > last_best) {
      // Validate sequence
      AdvState st = INIT_ST;
      int dmg = 0;
      int frames = 0;
      for (auto c : best_seq) {
        switch (c) {
          case 'x':
            switch (st.s.combo_) {
              case NO_COMBO:
                frames += COMBO_STARTUP;
                break;
              case AFTER_FS:
                frames += FS_RECOVERY + COMBO_STARTUP;
                break;
              default:
                frames += COMBO_RECOVERY[st.s.combo_];
                if (st.s.combo_ == AFTER_C5) {
                  frames += COMBO_STARTUP;
                }
                break;
            }
            dmg += COMBO_DMG[next_combo_state(st.s.combo_)];
            st.s.combo_ = next_combo_state(st.s.combo_);
            break;
          case '1':
            frames += SKILL_STARTUP;
            frames += SKILL_FRAMES[0];
            dmg += SKILL_DMG[0];
            st.s.combo_ = NO_COMBO;
            break;
          case '2':
            frames += SKILL_STARTUP;
            frames += SKILL_FRAMES[1];
            dmg += SKILL_DMG[1];
            st.s.combo_ = NO_COMBO;
            break;
          case 'f':
            dmg += FS_DMG;
            switch (st.s.combo_) {
              case NO_COMBO:
                frames += FS_STARTUP;
                break;
              case AFTER_FS:
                frames += FS_STARTUP + FS_RECOVERY;
                break;
              default:
                frames += XFS_STARTUP[st.s.combo_];
                break;
            }
            st.s.combo_ = AFTER_FS;
            break;
          default:
            std::cerr << "Invalid action: " << c << "\n";
            return -1;
        }
      }
      if (dmg != best || frames != f) {
        std::cerr << "VALIDATION FAILED: validator says "
            << dmg << " in " << frames << " but optimizer thought "
            << best << " in " << f << "\n";
        return -1;
      }
      std::cerr << best_seq << " for " << best << " dmg in " << f << " frames\n";
      last_best = best;
    }
  }

  return 0;
}
