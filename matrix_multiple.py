n=int(input())
a=[]
b=[]
for i in range(n):
    a.append(list(map(int,input().split())))
for i in range(n):
    b.append(list(map(int,input().split())))
c=[[0 for j in range(n)]for i in range(n)]
for i in range(n):
    for j in range(n):
        p=0
        for k in range(n):
           p+=a[i][k]*b[k][j]
        c[i][j]=p
for i in range(n):
    for j in range(n):
        print(c[i][j],end=' ')
    print()
