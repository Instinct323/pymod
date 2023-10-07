import re
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree

BASE_TYPE = int, float, bool


def load_tree(file: str):
    ''' :return: 数据树根结点 (可迭代)
            tag, text, attrib, tail'''
    assert re.search(r'(\.xml)$', file)
    return ElementTree(file=file).getroot()


def dump_tree(data, parent: Element = None):
    ''' 创建 xml 结构树
        :param data: 需要存储的数据
        :param parent: 双亲结点'''
    # 默认 parent 为 None，创建 root
    if parent is None: parent = Element('root')
    if isinstance(data, dict):
        data, fold_flag = data.copy(), True
        for k, v in data.items():
            # 检查 key 的数据类型
            if isinstance(k, BASE_TYPE):
                raise TypeError(f'Invalid key type: {type(k).__name__}')
            # 检查并转换 value 的类型
            if isinstance(v, (list, tuple)):
                fold_flag = False
            elif isinstance(v, BASE_TYPE) or v is None:
                data[k] = str(v)
        # 当 dict 中存在 list、tuple, 取消折叠
        if fold_flag:
            parent.attrib = data
        # 每个 key 都创建对应的结点
        else:
            for k in data: dump_tree(data[k], parent=SubElement(parent, k))
    # 每个 element 都创建对应的结点
    elif isinstance(data, (list, tuple)):
        for v in data: dump_tree(v, parent=SubElement(parent, parent.tag))
    # 直接添加内容
    elif isinstance(data, str):
        parent.text = data
    else:
        raise TypeError(f'Invalid data type: {type(data).__name__}')
    return parent


def dumps(data, file: str = None, encoding='utf-8'):
    ''' 创建xml文本流
        :param data: 需要存储的数据
        :param file: 文件名称 (无需后缀)
        :return: xml文本'''
    doc = parseString(tostring(dump_tree(data)))
    xml_str = doc.toprettyxml(encoding=encoding).decode(encoding)
    if file:
        assert re.search(r'(\.xml)$', file)
        with open(file, 'w', encoding=encoding) as f: f.write(xml_str)
    return xml_str


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
    print(dumps(data))
