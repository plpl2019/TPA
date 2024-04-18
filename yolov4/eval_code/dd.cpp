// #include <iostream>
// // #include <stdio.h>
// using namespace std;
// // // int func(int x)
// // // {
// // //     int count=0;
// // //     while(x)
// // //     {
// // //         count++;
// // //         x=x&(x-1);

// // //     }
// // //     return count;
// // // }
// // void print(int value){
// //     if(value != 0){
// //         for(int i=1;i<=value;i++){
// //             printf("%d",value);
// //         }
// //         print(value -1);
// //     }
// // }
// // int main(){
// //     // unsigned int time = 0x80402008;
// //     // printf("%x,%x,%x\n",(char)time,*((char*)&time+1),*((char*)&time+2));
// //     std::cout<<print(3)<<endl;
// //     return 0;
// // }
// class W{
//     public:
//         W(int i):s(i){}
//         int get(){return s;}
//         W operator*(int a){return W(2*s*a);}
//     private:
//         int s;

// };
// int main()
// {
//     W w1(3);
//     cout<<(w1*3).get()<<endl;
//     return 0;
// }

// #include<iostream>
// #include<algorithm>
// using namespace std;

// int dp[31][31][31][31] = { 0 };

// int main()
// {
//     int n;
//     int value[31][31] = { 0 };
//     cin >> n;
//     for (int i = 1; i <= n; i++)
//         for (int j = 1; j <= n; j++)
//         {
//             cin >> value[i][j];
//         }
//     for (int i = 1; i <= n; i++)
//     {
//         for (int j = 1; j <= n; j++)
//         {
//             for (int h = 1; h <= n; h++)
//             {
//                 for (int k = 1; k <= n; k++)
//                 {
//                     if (i == h && j == k)
//                     {
//                         dp[i][j][h][k] = max(dp[i - 1][j][h - 1][k], dp[i - 1][j][h][k - 1]);
//                         dp[i][j][h][k] = max(dp[i][j][h][k], dp[i][j - 1][h - 1][k]);
//                         dp[i][j][h][k] = max(dp[i][j][h][k], dp[i][j - 1][h][k - 1]);
//                         dp[i][j][h][k] += value[i][j];
//                     }
//                     else {
//                         dp[i][j][h][k] = max(dp[i - 1][j][h - 1][k], dp[i - 1][j][h][k - 1]);
//                         dp[i][j][h][k] = max(dp[i][j][h][k], dp[i][j - 1][h - 1][k]);
//                         dp[i][j][h][k] = max(dp[i][j][h][k], dp[i][j - 1][h][k - 1]);
//                         if(value[i][j]!=0&& value[h][k]!=0) dp[i][j][h][k] += 2;
//                             else if(value[i][j] != 0)  dp[i][j][h][k] += 1;
//                             else if(value[h][k] != 0)  dp[i][j][h][k] += 1;
//                     }
//                 }
//             }
//         }
//     }
//     cout << dp[n][n][n][n] << endl;
//     return 0;
// }

// #include <bits/stdc++.h>
// #define rep(i,a,b) for(int i=a;i<=b;++i)
// #define pb push_back
// typedef long long ll;
// const int N = 1e5 + 7;
// using namespace std;
// int n;
// vector<int> g[N];
// ll up[N],down[N],a[N];//up[u]: 至少加多少次， down[u]: 至少减多少次可以消去u这棵子树
// void dfs(int u,int pre) {
//     for(auto v: g[u]) if(v!=pre){
//         dfs(v,u);
//         up[u] = max(up[u],up[v]);
//         down[u] = max(down[u],down[v]);
//     }
//     a[u] = a[u] - down[u] + up[u];
//     if(a[u]>=0) down[u] += a[u];
//     else up[u] += abs(a[u]);
// }
// int main() {
//     scanf("%d",&n);
//     int x,y;
//     rep(i,1,n-1) scanf("%d%d",&x,&y),g[x].pb(y),g[y].pb(x);
//     rep(i,1,n) scanf("%I64d",&a[i]);
//     dfs(1,0);
//     printf("%I64d\n",up[1]+down[1]);
//     return 0;
// }

// #include <bits/stdc++.h>
// using namespace std;
// vector<vector<int> >res;
// vector<int>path;
// void dfs(vector<int>&fee,vector<int>&dis,int cur,int cost,int ability) {
//     if (accumulate(dis.begin()+cur,dis.end(),0)<=ability) {
//         res.push_back(path);
//         return ;
//     }
//     for (int j = cur + 1; j < fee.size(); j++){
//         if (accumulate(dis.begin() + cur, dis.begin() + j, 0) <= ability) {
//             path.push_back(j);
//             dfs(fee, dis, j, cost+fee[cur],ability);
//             path.pop_back();
//         }
//         else break;
//     }
// }
// int calFee(vector<int>r,vector<int>fee){
//     int sum;
//     for(int i=0;i<r.size();i++){
//         sum+=fee[r[i]];
//     }
//     return sum;
// }
// int main(){
//     int n;
//     cin>>n;
//     vector<int>fee(n);
//     vector<int>dis(n);
//     for(int i=0;i<n;i++){
//         cin>>fee[i];
//     }
//     for(int i=0;i<n;i++){
//         cin>>dis[i];
//     }
//     int ability;
//     cin>>ability;
//     path.push_back(0);
//     dfs(fee,dis,0,0,ability);
//     int totalFee=INT_MAX;
//     vector<int>road;
//     if(res.size()==0)cout<<-1;
//     for(auto r:res){
//         if(calFee(r,fee)<totalFee){
//             road=r;
//             totalFee= calFee(r,fee);
//         }
//     }
//     for(auto _road:road){
//         cout<<_road<<' ';
//     }

// } 

#include <bits/stdc++.h>

using namespace std;
int res=INT_MAX;
int dis[4][2]={{-1,0},{1,0},{0,1},{0,-1}};
void dfs(int x,int y,vector<vector<int>>&height,vector<vector<bool>>&isUsed,int rest,int count,int t){
    if(rest<0)return ;
    if(x==height.size()-1&&y==height[0].size()-1&&rest>=0){
        res=min(res,count);
        return ;
    }
    int curHeight=height[x][y];
    for(int i=0;i<4;i++){
        int nextX=x+dis[i][0];
        int nextY=y+dis[i][1];
        if(nextX>=0&&nextX<height.size()&&nextY>=0&&nextY<height[0].size()&&!isUsed[nextX][nextY]){
            isUsed[nextX][nextY]=true;
            if(abs(height[nextX][nextY]-curHeight)>t){
                dfs(nextX,nextY,height,isUsed,rest-1,count+1,t);
                isUsed[nextX][nextY]= false;
            }
            else{
                dfs(nextX,nextY,height,isUsed,rest,count+1,t);
                isUsed[nextX][nextY]= false;
            }
        }
    }
}
int main (){
    int m,n,t;
    cin>>m>>n>>t;
    vector<vector<int>>height(m,vector<int>(n,0));
    vector<vector<bool>>isUsed(m,vector<bool>(n,false));
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            cin>>height[i][j];
        }
    }
    isUsed[0][0]=true;
    dfs(0,0,height,isUsed,3,0,t);
    if(res==INT_MAX)cout<<-1;
    else{
        cout<<res;
    }
    return 0;
}