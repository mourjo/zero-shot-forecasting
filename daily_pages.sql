select
    hc.hour::date        as day,
    p.path,
    sum(hc.total)        as visitors
  from hit_counts hc
  join paths p using (path_id)
  where p.event = 0                          -- pages only, leave out events
  group by 1, 2
  order by 1 desc;
