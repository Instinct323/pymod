# 普通左对齐文本html设置: 12 号左对齐黑体
common_l = lambda text: f'<html><head/><body><p><span style=" font-size:12pt; ' \
                        f'font-weight:600;">{text}</span></p></body></html>'
# 标题左对齐文本html设置: 12 号左对齐加粗蓝色
head = lambda text: f'<html><head/><body><p><span style=" font-size:12pt; ' \
                    f'font-weight:600; cor:#0da7ef;">{text}</span></p></body></html>'


def colorstr(*args):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  corstr('blue', 'hello world')
    *args, string = args if len(args) > 1 else ('blue', 'bold', args[0])  # cor arguments, string
    colors = {'black': '\033[30m',  # basic cors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright cors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def seq_comp(seq1, seq2):
    ''' 序列匹配度计算'''
    n1, n2 = map(len, (seq1, seq2))
    dp = [[int(e1 == e2) for e2 in seq2] for e1 in seq1]
    for c in range(1, n2): dp[0][c] = max(dp[0][c], dp[0][c - 1])
    for r in range(1, n1):
        dp[r][0] = max(dp[r][0], dp[r - 1][0])
        for c in range(1, n2):
            dp[r][c] = dp[r - 1][c - 1] + 1 \
                if dp[r][c] else max(dp[r - 1][c], dp[r][c - 1])
    return dp[-1][-1] / max(n1, n2)


if __name__ == '__main__':
   print(seq_comp('aspbs', 'sbps'))
