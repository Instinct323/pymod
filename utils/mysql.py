import pymysql

conn = pymysql.connect(host='localhost',
                       user='root',
                       password='20010323',
                       database='library_3120000881')
cur = conn.cursor()

if __name__ == '__main__':
    class timer:

        def __init__(self, repeat: int = 1, avg: bool = True):
            self.repeat = max(1, int(repeat) if isinstance(repeat, float) else repeat)
            self.avg = avg

        def __call__(self, func):
            import time

            def handler(*args, **kwargs):
                t0 = time.time()
                for i in range(self.repeat): func(*args, **kwargs)
                cost = (time.time() - t0) * 1e3
                return cost / self.repeat if self.avg else cost

            return handler


    @timer(1e2)
    def test(*queries):
        for q in queries:
            cur.execute(q)


    test(r'''select category
    from borrow
             left join book on book.bno = borrow.bno
    group by category
    order by count(*) desc
    limit 0, 1;''')

    test(r'''create temporary table tmp as (select category, count(*) cnt
                     from borrow
                              left join book on book.bno = borrow.bno
                     group by category);''',
         r'''select category
    from tmp
    where cnt = (select max(cnt) from tmp);''',
         r'''drop table tmp;''')
