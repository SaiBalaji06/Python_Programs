grid=[]
for i in range(1):
    grid.append(list(map(int,input().split())))
mc=0
for i in range(len(grid)+1):
    m=[]
    for j in range(len(grid)):
        m.append(max(grid[j]))
        a=grid[j].index(max(grid[j]))
        grid[j].pop(a)
    mc=mc+max(m)
print(mc)
