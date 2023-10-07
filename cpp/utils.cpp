#include <fstream>
#include <iostream>

using namespace std;


/*
 *  @brief  控制文件的输入和输出
 *  @tparam dType   要序列化/反序列化的数据类型
 */
template<class dType>
class Pickle {

protected:
    string file;

public:
    // @param file  要与 Pickle 对象关联的文件路径。
    explicit Pickle(const string &file) : file(file) {}

    // 打开文件
    fstream open(ios::openmode mode) {
        fstream f(this->file, mode | ios::binary);
        // 检查文件打开状态
        if (!f)
            cerr << "Failed to open file." << endl;
        return f;
    }

    // 序列化
    void dump(const dType &obj) {
        fstream f = this->open(ios::out);
        if (f) {
            f.write((const char *) &obj, sizeof(obj));
            f.close();
        }
    }

    // 反序列化
    void load(dType &obj) {
        fstream f = this->open(ios::in);
        if (f) {
            f.read((char *) &obj, sizeof(obj));
            f.close();
        }
    }

    friend ostream &operator<<(ostream &os, const Pickle &obj) {
        os << "Pickle<" << obj.file << ">";
        return os;
    }
};


class Tmp {

public:
    int b;
    int a;

    Tmp(int a) : a(a) {}
};


int main() {
    Pickle<Tmp> pkf("data.txt");

    Tmp origin(1);
    pkf.dump(origin);

    Tmp current(32);
    pkf.load(current);
    cout << current.a;

    return 0;
}
