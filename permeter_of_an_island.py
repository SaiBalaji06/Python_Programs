#finding the permeter of an island (since 0-water,1-land) and there is only one island
n=int(input())
lst=[]
for i in range(n):
    lst.append(list(map(int,input().split())))
p=0
N=len(lst)
m=len(lst[0])
for i in range(N):
    for j in range(m):
        if lst[i][j]==1:
            if i==0 or lst[i-1][j]==0 :        #top(if top==0)
                p=p+1
            if j==0 or lst[i][j-1]==0:        #left(if left==0)
                p=p+1
            if j==(m-1) or lst[i][j+1]==0 or j==m-1:    #right(if right==0)
                p=p+1
            if i==(N-1) or lst[i+1][j]==0:    #bottom(if bottom==0)
                p=p+1
print(p)
