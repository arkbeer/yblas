#pragma once
#include<random>
#include<algorithm>
namespace ark {
	class random {
		std::mt19937 mt;
	public:
		random() {
			std::random_device rnd;
			mt = std::mt19937(rnd());
		}
		const int range_int(const int x, const int y) {
			auto it = std::minmax(x, y);
			std::uniform_int_distribution<> rand_(it.first, it.second);
			return rand_(mt);
		}
        const double range_real(const double x, const double y) {
			auto it = std::minmax(x, y);
			std::uniform_real_distribution<double> rand_(it.first, it.second);
			return rand_(mt);
		}
        const auto rand(){
            return mt();
        }
	};
}