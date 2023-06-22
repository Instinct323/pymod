def xpath(root, value, verbose=False):
    ''' selenium 结点元素定位
        root: 根结点
        value: xpath 表达式
        verbose: 输出调试信息'''
    while 1:
        try:
            result = root.find_elements('xpath', value)
            if result:
                return result if len(result) != 1 else result[0]
        except:
            if verbose: print('\r未找到相应元素...', end='')
