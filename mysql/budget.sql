# show variables like 'datadir';
drop database budget;
create database budget charset utf8;
use budget;

# 每月收支
create table monthly
(
    start  char(7) comment '开始日期',
    end    char(7) comment '截止日期',
    value  decimal(10, 2) comment '收支 (k/month)',
    detail char(30) comment '详细说明' not null,

    constraint s check (start regexp '\\d{4}-\\d{2}'),
    constraint e check (end regexp '\\d{4}-\\d{2}')
);

# 收入项
insert into monthly
values ('2024-03', '2024-07', 0.4, '贫困补贴'),
       ('2024-03', '2024-08', 4.0, '实验室补贴'),

       ('2024-09', '2025-08', 2 + 0.5 + 0.6, '基本补贴'),
       ('2024-12', '2024-12', 4.5, '学业奖学金'),
       ('2024-09', '2025-06', 1.0, '助教补贴'),

       ('2025-09', '2026-08', 2 + 0.5 + 0.6, '基本补贴'),
       ('2025-12', '2025-12', 4.5, '学业奖学金'),
       ('2025-09', '2026-06', 1.0, '助教补贴'),

       ('2026-09', '2027-08', 2 + 0.5 + 0.6, '基本补贴'),
       ('2026-12', '2026-12', 4.5, '学业奖学金');

# 支出项
insert into monthly
values ('2024-03', '2024-08', -0.7, '房租 (e.g., 民治五和城中村)'),
       ('2024-09', '2026-09', -9.5 / 12, '南科大学费');

select *
from monthly
# where '2024-03' between start and end
order by start, end, value;
