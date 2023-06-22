import re
from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree
from xml.dom.minidom import parseString

BASE_TYPE = (int, float, bool)


def dict_tran(dict_obj: dict):
    ''' 创建纯字符的字典
        return: new_dict, fold_flag'''
    new_dict = {}
    fold_flag = True
    for key in dict_obj:
        value = dict_obj[key]
        if isinstance(value, (list, tuple)):
            fold_flag = False
        elif isinstance(value, BASE_TYPE):
            value = str(value)
        elif value is None:
            value = 'None'
        if isinstance(key, BASE_TYPE):
            raise TypeError(f'不可以 {BASE_TYPE} 数据类型作字典的键')
        new_dict[key] = value
    return new_dict, fold_flag


def dump_tree(data, parent: Element = None):
    ''' 创建xml结构树
        data: 需要存储的数据
        parent: 双亲结点'''
    if parent is None:
        parent = Element('root')
        # 默认parent为None，创建root
    if isinstance(data, dict):
        data, fold_flag = dict_tran(data)
        # 当dict中存在list、tuple，取消折叠
        if fold_flag:
            parent.attrib = data
            # 直接设置属性
        else:
            for key in data:
                value = data[key]
                leaf = SubElement(parent, key)
                dump_tree(value, parent=leaf)
                # 每个key都创建对应的结点
    elif isinstance(data, (list, tuple)):
        for value in data:
            member = parent.tag
            leaf = SubElement(parent, member)
            dump_tree(value, parent=leaf)
            # 每个element都创建对应的结点
    elif isinstance(data, str):
        parent.text = data
        # 直接添加内容
    else:
        raise TypeError('出现未能识别的数据类型')
    return parent


def dump(data, file: str = None, encoding='utf-8'):
    ''' 创建xml文本流
        data: 需要存储的数据
        file: 文件名称 (无需后缀)
        xml_str: xml文本'''
    root = dump_tree(data)
    doc = parseString(tostring(root))
    xml_str = doc.toprettyxml(encoding=encoding).decode(encoding)
    if file:
        assert re.search(r'(\.xml)$', file)
        with open(file, 'w', encoding=encoding) as f:
            f.write(xml_str)
    return xml_str


def load(file: str):
    ''' return: 数据树根结点 (可迭代)
            tag, text, attrib, tail'''
    assert re.search(r'(\.xml)$', file)
    tree = ElementTree(file=file)
    root = tree.getroot()
    return root


if __name__ == '__main__':
    data = {'name': '知识树',
            'children':
                [{'name': '机器学习',
                  'children':
                      [{'name': 'ResNet: 复现分类卷积神经网络'},
                       {'name': 'Yolo: 会用输出张量做pred，理解loss函数的架构'},
                       {'name': 'Q-learning,Sarsa,DQN: 了解强化学习的基础方法'}]
                  },
                 {'name': '测试数据',
                  'children':
                      [{'bool': True, 'float': 12.3}]
                  }]
            }
    print(dump(data))