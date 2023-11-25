#pragma once

template<typename T>
void processArgs(T arg);


/**
 * @brief 递归处理不定数量的参数
 * @param first - 第一个参数
 * @param rest - 剩余的参数
 */
template<typename T, typename... Args>
void processArgs(T first, Args... rest) {
    processArgs(first);
    processArgs(rest...);
}
