#include <filesystem>
#include <iostream>

using namespace std;
namespace fs = std::filesystem;


void remove_if_empty(const fs::path root) {
    try {
        // 遍历目录下每一个文件夹
        for (const fs::path f: fs::directory_iterator(root)) {
            if (fs::is_directory(f)) {
                // 删除空文件夹
                remove_if_empty(f);
            }
        }
        // 文件夹被清空时, 删除文件夹, 并输出
        if (fs::is_empty(root)) {
            fs::remove(root);
            cout << "remove " << root << endl;
        }
    } catch (const exception &e) {
        // 捕获异常信息
        cout << "error " << e.what() << endl;
    }
}


int main() {
    remove_if_empty("C:\\");
    remove_if_empty("D:\\");

    return 0;
}
