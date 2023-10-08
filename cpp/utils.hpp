#pragma once

#include <iostream>
#include <fstream>

using namespace std;


/**
 * @brief 切换 Windows 命令行编码为 UTF-8
 * @see https://docs.microsoft.com/en-us/windows-server/administration/windows-commands/chcp
 */
void use_utf8() {
    system("chcp 65001");
}


/**
 *  @brief 控制对象的存储
 *  @tparam dType 要序列化/反序列化的数据类型
 */
template<class dType>
class BinFile {

public:
    /**
     * @param file 要与 BinFile 对象关联的文件路径
     */
    explicit BinFile(const string &file) : file(file) {}

    /**
     *  @brief 序列化函数, 用于将对象序列化并写入文件
     */
    void dump(const dType &obj) {
        fstream f = this->open(ios::out);
        if (f) {
            f.write((const char *) &obj, sizeof(obj));
            f.close();
        }
    }

    /**
     *  @brief 反序列化函数, 用于从文件中读取对象
     */
    void load(dType &obj) {
        fstream f = this->open(ios::in);
        if (f) {
            f.read((char *) &obj, sizeof(obj));
            f.close();
        }
    }

protected:
    string file;

    fstream open(ios::openmode mode) {
        fstream f(this->file, mode | ios::binary);
        // 检查文件打开状态
        if (!f)
            cerr << "Failed to open file." << endl;
        return f;
    }

    friend ostream &operator<<(ostream &os, const BinFile &obj) {
        os << "BinFile<" << obj.file << ">";
        return os;
    }

};
