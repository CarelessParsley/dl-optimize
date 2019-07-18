#include <iostream>
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>

// Erik
constexpr double STRENGTH = 2980.56;

constexpr int SKILL_SP[2] = {2868, 5883};
constexpr int SKILL_STARTUP = 6;
constexpr int SKILL_RECOVERY[2] = {111, 114};
constexpr int SKILL_MOD[2] = {1036, 933};

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
  NO_COMBO = -1,
  AFTER_C1 = 0,
  AFTER_C2 = 1,
  AFTER_C3 = 2,
  AFTER_C4 = 3,
  AFTER_C5 = 4,
  AFTER_S1 = 79,
  AFTER_S2 = 89,
  AFTER_FS = 99,
};

int handle_recovery(ComboState c, bool combo_recovery) {
  switch (c) {
    case AFTER_FS:
      return FS_RECOVERY;
    case AFTER_S1:
      return SKILL_RECOVERY[0];
    case AFTER_S2:
      return SKILL_RECOVERY[1];
    case NO_COMBO:
      return 0;
    default:
      if (combo_recovery) {
        return COMBO_RECOVERY[c];
      }
      return 0;
  }
}

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

std::ostream& operator<<(std::ostream& os, AdvState st) {
  os << "[sp=" << st.s.sp_[0] << "," << st.s.sp_[1] << ";c=" << st.s.combo_ << "]";
  return os;
}

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
    for (int i = 0; i < 2; i++) {
      if (s.s.sp_[i] >= SKILL_SP[i]) {
        AdvState t = s;
        t.s.sp_[i] = 0;
        t.s.combo_ = i == 0 ? AFTER_S1 : AFTER_S2;
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

int main(int argc, char** argv) {

  int frames;
  std::cerr << "frames? ";
  std::cin >> frames;
  std::cerr << frames << "\n";

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

  std::vector<double> best_dps(frames * num_states, -1);
  std::vector<std::string> best_sequence(frames * num_states);

  auto dix = [&](int frame, int state_ix) {
    return frame * num_states + state_ix;
  };

  best_dps[dix(0, state2ix.at(INIT_ST.c))] = 0;

  double last_best = 0;
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

        int mod = 0;
        if (ac == 'x') {
          frames += handle_recovery(p_st.s.combo_, /*combo_recovery*/ true);
          if (st.s.combo_ == AFTER_C1) {
            frames += COMBO_STARTUP;
          }
          mod = COMBO_MOD[st.s.combo_];
        } else if (ac == 'f') {
          // Force strike cancels combo recovery; instead,
          // there's specific startup costs in this case
          frames += handle_recovery(p_st.s.combo_, /*combo recovery*/ false);
          switch (p_st.s.combo_) {
            case NO_COMBO:
            case AFTER_S1:
            case AFTER_S2:
            case AFTER_FS:
              frames += FS_STARTUP;
              break;
            default:
              frames += XFS_STARTUP[p_st.s.combo_];
              break;
          }
          mod = FS_MOD;
        } else {
          // Skill cancels all recovery
          frames += SKILL_STARTUP;
          mod = SKILL_MOD[ac == '1' ? 0 : 1];
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
            if (ac == '1' || ac == '2') {
              dmg *= 1.4; // skill_ab, from KFM + FitF
              // dmg *= skill_buffs;
              // dmg *= skill_coab;
            } else if (ac == 'f') {
              dmg *= 1.3; // Erik's ability
            }
            // dmg *= punisher;
            // dmg *= elemental;
            dmg /= 10. * 1.;  // defense * defense change
            dmg *= (1. + (0.04 + 0.14 /*KFM*/) * (0.7 + 0.15 /* FitF */)); // crit rate * crit damage

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
        if (true) {
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
