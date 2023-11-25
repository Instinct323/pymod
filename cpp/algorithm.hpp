#pragma once

#include <algorithm>
#include <string>
#include <vector>

using namespace std;


/**
 *  @brief manacher 算法求最长回文串
 *  @var p - 以对应字符为中心 最长回文串的半径
 */
class Manacher {
public:
    string s;
    vector<int> p;

    inline bool is_same(int i0, int i1) {
        return (i0 >= 0) && (i1 < s.size()) && (s[i0] == s[i1]);
    }

    Manacher(string &_s) {
        // 对字符串进行预处理
        s.assign(2 * _s.size() + 2, '#');
        s[0] = '^';
        s.push_back('$');
        for (int i = 0; i < _s.size(); ++i) {
            s[2 * (i + 1)] = _s[i];
        }
        // 最长回文串的中心、边界
        int center = 0, border = 0;
        // 以对应字符为中心 最长回文串的半径
        p.assign(s.size(), 0);
        for (int i = 1; i < s.size(); ++i) {
            // 利用回文串的对称性进行赋值 (注意索引为正)
            p[i] = min(p[max(0, 2 * center - i)], max(0, border - i));
            // 中心扩展法
            while (is_same(i - p[i] - 1, i + p[i] + 1)) { ++p[i]; }
            // 更新回文串中心, 回文串右端点 ('#')
            if (i + p[i] > border) {
                center = i;
                border = i + p[i];
            }
        }
    }
};
