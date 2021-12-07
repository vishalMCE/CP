
/* <---------------------General Use-------------------> */
// count power of 2 in n
int power2(int n) {
    int pw = 0;
    while(n%2==0) {
        pw++;
        n/=2;
    }
    return pw;
}
int sumofdigit(int n) {
    int sum = 0;
    while(n!=0) {
        sum += n%10;
        n /= 10;
    }
    return sum;
}
// reverse number 
int rev(int n) {
    int ans=0;
    while(n!=0) {
        ans = ans*10 + n%10;
        n /= 10;
    }
    return ans;
}
bool ispalindrome(string &s) {
    string temp = s;
    reverse(all(temp));
    return s == temp;
}

// subsequence checking  -- O(n)
bool solve(string s, string t) {
    int n = s.size();
    int m = t.size();
    int i=0;
    for(int j=0; j<m && i<n; ++j)
        if(s[i]==t[j]) i++;
    return i==n;
}

// Couting number of 1-Bits from 1 to n   --   O(n)
vi solve(int n) {
    vector<int> ans(n+1);
    for(int i=1; i<=n; ++i)
        ans[i] = ans[i/2]+i%2;
    return ans;
}

// O(1.6180^N)
int fib(int n){
    if(n<=1) return n;
    return fib(n-2)+fib(n-1);
}
// Pascal Traiangle  --  O(n^2)
void pascal_triangle(int n) {
    vii v(n);
    f(i,0,n) f(j,0,i+1) {
        if(i>1 && j!=0 && j!=i) v[i].pb(v[i-1][j-1]+v[i-1][j]);
        else v[i].pb(1);
    }
    return v;
}
// generate random number within range
vi gen_rand(int lower, int upper, int n) {
    vi ans(n); srand(time(0));
    f(i,0,n) ans[i]=(rand()%(upper-lower+1)+lower);
    return ans;
}
// input comma seperateed values 
void comma(string s, vi& v) {
    stringstream ss(s);
    for(int i; ss>>i;) {
        v.pb(i);
        if(ss.peek()==',') ss.ignore();
    }
}




/* <---------------------Array Problems-------------------> */

// Maximum subarray problem   -  O(n^2)
int Maximum_subarray_sum(vi v){
    int n = v.size();
    int ans = INT_MIN;
    f(i,0,n) {
        int sum=0;
        f(j,i,n) {
            sum+=v[j];
            ans = max(sum,ans);
        }
    }
    return ans;
}
// Kadane's Algorithms - O(n) 
int Maximum_subarray_sum(vi v){
    int n = v.size();
    int ans = INT_MIN,sum=0;
    for(auto x: v){
        sum = max(x,x+sum);
        ans = max(ans,sum);
    }
    return ans;
}
// maximum product subarray  Kadane's Algo  -  O(n)
int solve(vi v) {
    int n = v.size();
    int ans = v[0], maxi=v[0], mini=v[0]; 
    for(int i=1; i<n; ++i){
        if(v[i]<0) swap(maxi,mini);
        maxi = max(v[i],v[i]*maxi);
        mini = min(v[i],v[i]*mini);
        ans = max(ans,maxi);
        // cout<<maxi<<" "<<ans<<endl;
    }
    return ans;
}
// find maximum subarray also indices   -  O(n)
int a=-1, b=-1;
int solve(vi v){
    int n = v.size();
    int ans = INT_MIN,sum=0; int j=0;
    f(i,0,n) {
        sum += v[i];
        if(ans<sum) ans=sum, a=j, b=i; 
        if(sum<0) sum=0, j=i+1;
    }
    return ans;
}
// subset sum problem 
vi ans;
bool solve(vi v, int index, int x) {
    if(x == 0) return true;
    for(int i = index; i<v.size(); ++i) {
        if(v[i]<=x && solve(v,i+1,x-v[i])){    
            ans.pb(v[i]);
            return true;
        }
    }
    return false;
}
// subarray with given sum  -- O(n)  | only positive numbers 
vi solve(vi v, int x) {
    int n = v.size();
    int l=0, h=0, sum=v[0];
    while(l<n && h<n) {
        if(sum==x) return {l,h};
        if(sum<x) { ++h; if(h<n) sum+=v[h]; }
        else sum-=v[l++];
    }
    return {-1};
}
// subarray with given sum  --  O(n) 
vi solve(vi v, int x) {
    int n = v.size();
    unordered_map<int,int>m;
    int sum=0;
    f(i,0,n) {
        sum+=v[i];
        if(sum==x) return {0,i};
        if(m.find(sum-x)!=m.end()) return {m[sum-x]+1,i};
        m[sum]=i;
    }
    return {-1};
}
// count number of subarray with given sum  --   O(n)
int solve(vi v, int x) {
    int n = v.size();
    unordered_map<int,int>m;
    int sum=0, ans=0; 
    f(i,0,n) {
        sum+=v[i];
        if(sum==x) ans++;
        if(m.find(sum-x)!=m.end()) ans+=m[sum-x];
        m[sum]++;
    }
    return ans;
}
// Generate all Parentheses
void gen_paran(string s, int i, int j) { 
    if(i==0 && j==0) { ans.pb(s); return; }
    if(i>0) gen_paran(s+'(', i-1, j+1);
    if(j>0) gen_paran(s+')', i, j-1);  
}





/* <---------------------Complete Search-------------------> */

// generating subsets recursive method -- O(2^N)
void subsetsUtil(vi& A,vii& res,vi& subset,int index){ 
    res.pb(subset); 
    for (int i = index; i < A.size(); i++) {  
        subset.pb(A[i]);
        subsetsUtil(A, res, subset, i + 1); 
        subset.pop_back(); 
    } 
} 
vii subsets(vi& A) { 
    vi subset; 
    vii res;
    int index = 0; 
    subsetsUtil(A,res,subset,index);
    return res;
}
// Subset Generation using Bit Masking -- O(N*2^N)
vii getSubsets(vi num){
    int size = num.size();
    int subsetNum = (1<<size);
    vii allSubsets;
    for(int subsetMask = 0; subsetMask<subsetNum; ++subsetMask){
        vi subset;
        for(int i=0; i<size; ++i){
            if((subsetMask & (1<<i))!=0) subset.pb(num[i]);
        }
        allSubsets.pb(subset);
    }
    return allSubsets;
}

// generate permutation of distinct numbers
// template<class X>
// void solve(X v,vector<X> &ans,int x){
//     if (x == v.size()) {
//         ans.pb(v);
//         return;
//     }
//     for (int i = x; i < v.size(); ++i) {
//         swap(v[i], v[x]);
//         solve(v,ans,x+1);
//     }
// }

// generate permutaion of any numbers   O(n!)
template<class X>
void solve(X v,vector<X>& ans,int x){
    if (x == v.size()) {
        ans.pb(v);
        return;
    }
    for (int i = x; i < v.size(); ++i) {
        if(i!=x && v[i]==v[x])
            continue;
        swap(v[i], v[x]);
        solve(v,ans,x+1);
    }
}
template<typename X>
vector<X> permute(X nums){
    sort(all(nums));
    vector<X> ans;
    solve<X>(nums,ans,0);
    return ans;
}

// nQueen
void print(vii &v) {cont++;
    f(i,0,v.size()) {
        output(v[i]); cout<<endl; 
    }
    cout<<endl;
}
void print_numbers(vii &v) {cont++;
    cout<<"[";
    f(i,0,v.size()) {
        f(j,0,v.size()) { 
            if(v[i][j]==1){
                cout<<j+1<<" ";
            }
        }
    }
    cout<<"] ";
}
bool isSafe(vii &v, int x, int y, int n) {
    for(int row = 0; row < x; ++row) {
        if(v[row][y] == 1)
            return false;
    }
    int row = x; 
    int col = y;
    while(row >= 0 && col >= 0){
        if(v[row][col] == 1) 
            return false;
        row--;
        col--;
    }
    row = x;
    col = y;
    while(row >= 0 && col < n){
        if(v[row][col] == 1)
            return false;
        row--;
        col++;
    }
    return true;
}
bool nQueen(vii &v, int x, int n) {
    if(x >= n) {
        return true;
    }
    for(int col = 0; col < n; ++col) { 
        if(isSafe(v, x, col, n)) {
            v[x][col] = 1;
            if(nQueen(v, x+1, n)) {
                return true;
            }
            v[x][col] = 0;
        }
    }
    return false;
}

// generate all subarray of array  --  O(n*n) or n*(n+1)/2
void getsubarray(vi v, int idx, vi s, vii& ans, int& ct) { cnt++;
    if(idx>0) ans.pb(s);
    if(idx+1==v.size()) ct++;
    for(int i=idx; i<v.size(); ++i) {
        s.pb(v[i]);
        getsubarray(v,i+1,s,ans,ct);
        s.clear();
        if(i>=ct) break;
    }
}
vii subarray(vi v) {
    vii ans; 
    vi s; 
    int x=0;
    getsubarray(v,0,s,ans,x);
    return ans;
}






/* <-------------------Dynamic Programming-----------------> */

//          coins are sorted

// Minimum digit to sum  --  O(nk)
int solve(vi coins,int x) { cnt++;
    if(x==0) return 0;
    if(dp[x]) return dp[x];
    int best = inf;
    for(int c: coins) { 
        if(x>=c) best=min(best,solve(coins,x-c)+1);
        else break;
    }
    return dp[x]=best;
}
// iterative approach or bottoom up
int solve(vi& v, int x) {
    f(i,1,x+1) {
        dp[i]=INT_MAX;
        for(int a: v) {
            if(i>=a) dp[i]=min(dp[i], dp[i-a]+1);
            else break;
        }
    }
    return dp[x];
}
// Counting the number of solutions
int Count(vi v, int x) { cnt++;
    if(x==0) return 1;
    if(dp[x]) return dp[x];
    for(int c: v) {
        if(x>=c) dp[x]+=Count(Countv,x-c);
        else break;
    }
    return dp[x];
}
// iterative approach or bottoom up
int solve(vi& v, int x) {
    dp[0]=1;
    f(i,1,x+1) {
        for(int a: v) {
            if(i>=a) dp[i]+=dp[i-a];
            else break;
        }
    }
    return dp[x];
}

// Longest increasing subsequence   --   O(n*2)
int lis(vi v, int n) {
    vi len(n,1);
    f(k,0,n) f(i,0,k) {
        if(v[i]<v[k]) len[k] = max(len[k],len[i]+1);
    }
    sort(all(len),greater<>());
    return len[0];
}
// lis   --  O(n*long(n))
vi lis(vi v, int n) {
    vi seq; vi ans(n); ans[0]=1;
    seq.pb(v[0]);
    f(i,1,n) {
        if(seq.back()<v[i]){ seq.pb(v[i]);ans[i]=seq.size();continue;}
        int idx = lower_bound(all(seq),v[i]) - seq.begin();
        seq[idx] = v[i]; ans[i]=idx+1;
    }
    return ans;
}

// Path in a Grid 
int solve(vii v, int i, int j, int n) { cnt++;
    if(i==n-1 && j==n-1) return v[i][j];
    if(dp[i][j]) return dp[i][j];
    if(j+1<n) dp[i][j]=v[i][j]+solve(v,i,j+1,n);
    if(i+1<n) dp[i][j]=max(dp[i][j],v[i][j]+solve(v,i+1,j,n));
    return dp[i][j];
}
// more efficient
int solve(vii v) {
    int n = v.size();
    int m = v[0].size();
    dp[0][0]=v[0][0];
    f(i,0,n) {
        f(j,0,m) { 
            if(j) dp[i][j]=max(dp[i][j],dp[i][j-1]+v[i][j]);
            if(i) dp[i][j]=max(dp[i][j],dp[i-1][j]+v[i][j]);
        }
    }
    return dp[n-1][m-1];
}

// 0-1 Knapscak  -  recursive dp   --   O(n*w)
int solve(vi v, vi s, int w, int idx) { cnt++;
    if(w==0 || idx==v.size()) return 0;
    if(dp[idx][w]) return dp[idx][w];
    f(i,idx,v.size()) {
        if(w>=s[i]) 
            dp[idx][w]=max(dp[idx][w],solve(v,s,w-s[i],i+1)+v[i]);
    }
    return dp[idx][w];
} // v-> value, s->weight, idx=0, w=capacity


// Edit distance  -  recursive approach
int solve(int i, int j, string& s, string& t) {
    if(i==s.size() || j==t.size()) 
        return max(t.size()-j, s.size()-i);
    if(dp[i][j]) return dp[i][j];
    if(s[i]==t[j]) dp[i][j]=solve(i+1,j+1,s,t);
    else {
        int op1 = 1 + solve(i,j+1,s,t);  //add
        int op2 = 1 + solve(i+1,j,s,t);  //remove
        int op3 = 1 + solve(i+1,j+1,s,t);  //replace
        dp[i][j] = min(op1,min(op2,op3));
    }
    return dp[i][j];
}

// Efficient code of Edit distance
int dp[5001][5001];
int solve(string s, string t) {
    int n = s.size();
    int m = t.size(); 
    f(i,1,n+1) dp[i][0]=i;
    f(i,1,m+1) dp[0][i]=i;
    
    f(i,1,n+1) {
        f(j,1,m+1) {
            if(s[i-1]==t[j-1]) dp[i][j]=dp[i-1][j-1];
            else dp[i][j]=min({dp[i-1][j-1],dp[i-1][j],dp[i][j-1]})+1;
        }
    }
    return dp[n][m];
}

// job scheduling   --   O(n*log(n))
int jobScheduling(vi sT, vi eT, vi profit) {
    int n = sT.size();
    vii jobs; f(i,0,n) jobs.pb({eT[i],sT[i],profit[i]});
    sort(all(jobs)); printmat(jobs,jobs.size());
    map<int,int>dp{{0,0}};
    for(auto job: jobs) {
        int cur = prev(dp.upper_bound(job[1]))->ss+job[2];
        if(cur>dp.rbegin()->ss) dp[job[0]+1]=cur;
        // for(auto x: dp) cout<<x.ff<<" "<<x.ss<<endl;
    }
    return dp.rbegin()->ss;
}





/* <--------------------Amortized Analysis------------------> */

// subarray sum using 2-pointer  -- O(2*n)
pi solve(vi v, int x) {
    int n = v.size();
    int l = 0, r = 0, sum=v[0];
    while(l<n && r<n) {
        if(sum==x) return {l,r};
        else if(sum<x) { ++r; if(r<n) sum+=v[r]; }
        else if(sum>x) sum-=v[l++]; 
    }
    return {-1,-1};
}

// 2-SUM using 2-pointer  --  O(n*log(n)) 
vector<pi> twosum(vi v, int k) {
    int n = v.size();
    sort(all(v));
    int i=0, j=n-1;
    vector<pi> ans;
    while(i<j) {
        if(v[i]+v[j] == k) ans.pb({v[i],v[j]}),i++,--j;
        else if(v[i]+v[j] < k) i++;
        else --j;
    }
    return ans;
}
// 2-SUM using hashing  --  O(n) 
vector<pi> twosum(vi v, int sum) {
    int n = v.size();
    unordered_map<int,bool> m;
    vector<pi> ans;
    f(i,0,n) {
        int temp=sum-v[i]; 
        if(m.find(temp)!=m.end()) ans.pb({v[i],temp});
        m[v[i]]=true;
    }
    return ans;
}
unordered_set<int> st;
f(i,0,n) {
    if(st.find(s-v[i])!=st.end()) {
        cout<<s-v[i]<<" "<<v[i];
        break;
    }
    st.insert(v[i]);
}
// 3-SUM using 2-pointer  --  O(n^2)
vii threeSum(vi &v, int k) {
    sort(all(v));
    vii ans;
    set<vector<int>> s;
    int n = v.size();
    f(i,0,n-2) {
        int val = v[i];
        int l = i+1;
        int r = n-1;
        while(l<r){
            int sum = val + v[l] + v[r];
            if(sum == k) s.insert({val,v[l++],v[r--]});
            else if(sum<k) l++;
            else r--;
        }
    }
    for(auto x: s) ans.pb(x);
    return ans;
}
// 4-Sum    -    O(n^3)
vii solve(vi v, int k) {
    sort(all(v));
    vii ans;
    set<vector<int>> s;
    int n = v.size();
    f(i,0,n-3) {
        f(j,i+1,n-2) {
            int val = v[i];
            int val1 = v[j];
            int l = i+2;
            int r = n-1;
            while(l<r){
                int sum = val + v[l] + v[r] + val1;
                if(sum == k)s.insert({val,val1,v[l++],v[r--]});
                else if(sum<k) l++;
                else r--;
            }
        }
    }
    for(auto x: s) ans.pb(x);
    return ans;
}
// Nearest smaller elements  --  O(n)
vi prevSmaller(vi v) {
    int n = v.size();
    stack<int> s;
    vi ans(n);
    f(i,0,n) {
        int x = -1;
        while(!s.empty() && s.top()>=v[i]) s.pop();
        if(!s.empty()) x = s.top();
        ans[i] = x;
        s.push(v[i]);
    }
    return ans;
}
// Sliding window maximum  --   O(n)
vi swmax(vi& v, int k) {
    int n = v.size();
    deque<pi> dq;  
    vi ans;   
    f(i,0,n) {
        if(!dq.empty() && dq.front().ss<=(i-k))  
            dq.pop_front();
        while(!dq.empty() && dq.back().ff<v[i])
            dq.pop_back();
        dq.pb({v[i],i});
        if(i>=(k-1)) ans.pb(dq.front().ff);    
    }
    return ans;
}
// recursive DP
int lcs(string s, string t, int i, int j) { cnt++;
    if(i==s.size() || j==t.size()) return 0;
    if(dp[i][j]) return dp[i][j];
    else if(s[i]==t[j]) return dp[i][j]=1+lcs(s,t,i+1,j+1);
    return dp[i][j]=max(lcs(s,t,i+1,j),lcs(s,t,i,j+1));
}
// Longest Common Subsequence   --    O(n^2)
int lcs(string s, string t) {
    int n = s.size()+1;
    int m = t.size()+1;

    f(i,1,n) f(j,1,m) {
        if(s[i-1]==t[j-1]) dp[i][j]=1+dp[i-1][j-1];
        else dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
    }

    return dp[n-1][m-1];
}




/* <---------------------Range Queries-------------------> */

// minimum range query   --   O(n*long(n))
// Binary Indexed Tree or Fenwick Tree    --   O(n*long(n))
void update(vi& BIT, int index, int value, int n) {
    while(index<=n) {
        BIT[index]+=value;
        index+=index&(-index); 
    }
}
int query(const vi& BIT, int index) {
    int sum = 0;
    while(index>0) {
        sum+=BIT[index];
        index-=index&(-index);
    }
    return sum;
}
void assign(vi& BIT, int index, int value, int n) {
    int oldValue = BIT[index];
    while(index<=n) {
        BIT[index]-=oldValue;
        BIT[index]+=value;
        index+=index&(-index); 
    }
}
int range(const vi& BIT, int left, int right) { 
    return query(BIT,right-1)-query(BIT,left-1);
} // left or right is position


// Segment Tree [tree size will be 4n]
// O(n)
void buildTree(vi v, vi& tree, int start, int end, int treeNode) {
    if(start==end) { tree[treeNode] = v[start]; return; }
    int mid = (start+end)/2;
    buildTree(v,tree,start,mid,2*treeNode);
    buildTree(v,tree,mid+1,end,2*treeNode+1);
    tree[treeNode]=tree[2*treeNode]+tree[2*treeNode+1];
}
// log(n)
void updateTree(vi v, vi& tree, int start, int end, int treeNode, int idx, int value) {
    if(start==end) { tree[treeNode]=value; return; }
    int mid=(start+end)/2;
    if(idx>mid)updateTree(v,tree,mid+1,end,2*treeNode+1,idx,value);
    else updateTree(v,tree,start,mid,2*treeNode,idx,value);
    tree[treeNode]=tree[2*treeNode]+tree[2*treeNode+1];
}
// log(n)
int query(vi tree, int start, int end, int treeNode, int left, int right) {
    // completely outside given range 
    if(start>right || end<left) return 0;
    // completely inside given range
    if(start>=left && end<=right) return tree[treeNode];
    // partially inside and partially outside
    int mid = (start+end)/2;
    int ans1 = query(tree,start,mid,2*treeNode,left,right);
    int ans2 = query(tree,mid+1,end,2*treeNode+1,left,right);
    return ans1+ans2;
}






/* <---------------------Bit Manipultion-------------------> */
x&(1<<k)==1   --   check kth bit is one 
x|(1<<k)      --   set kth bit of x to one
x&~(1<<k)     --   set kth bit of x to zero
x^(1<<k)      --   invert kth bit of x
x&(x-1)       --   set last one bit of x to zero
x&(-x)     --   set all one bit to zero except the last one bit
x|(x-1)       --   invert all the bit after the last one bit
x&(x-1)==0    --   x is a power of two
range query   --
x&-x          --   find at most set bit. ex-10100->00100
x-(x&-x)      --   remove right most set bit    

// count the number of subgrid whose corner are black -- O(n^3)
int solve(vii v, int n) {
    int ans=0;
    f(i,0,n) {
        f(j,i+1,n) { int count=0;
            f(k,0,n) {
                if(v[i][k]==1 && v[j][k]==1) ++count;
            }
            ans+=((count-1)*count)/2;
        }
    }
    return ans;
}







/* <---------------------Number Theory--------------------> */

// gcd    --   O(log(max(a,b)))
int gcd(int a, int b) {
    if(b == 0) return a;
    return gcd(b, a%b);
}
// isprime     --   O(sqrt(n))    
bool isprime(int n) {
    if(n<=1) return false;
    for(int i=2; i*i<=n; ++i) {
        if(n%i==0) return false;
    }
    return true;
}
// prime factorization of a number  --  O(sqrt(n))
vi factors(int n) {
    vi ans;
    for (int x=2; x*x <= n; ++x) {
        while (n%x == 0) {
            ans.pb(x);
            n /= x;
        }
    }
    if(n > 1) ans.pb(n);
    return ans;
}
// sieve of eratosthenes  --  O(n*log(log(n))) ~ of(n)
vi SE(int n) {
    vi prime(n+1), primes;
    for(int x=2; x<=n; ++x) {
        if(prime[x]) continue;
        for(int u=2*x; u<=n; u+=x) prime[u] = x;
        primes.pb(x);
    }
    return primes;
}
// find all factors of n -- O(sqrt(n))
vi factors(int n) {
    set<int> s; s.insert(1); 
    for(int i=2; i*i<=n; ++i) {
        if(n%i==0) s.insert(i);
    }
    for(int x: s) s.insert(n/x);
    vi v(all(s));
    return v;
}
// Euler's totient function   --  O(sqrt(n))
int phi(int n) {
    int result=n;
    for(int i=2; i*i<=n; i++) {
        if (n%i==0) {
            while(n%i==0)
                n /= i;
            result -= result/i;
        }
    }
    if(n>1)
        result -= result/n;
    return result;
}
// modular exponentiation   --  O(log(n))
int modpow(int x, int n, int m) {
    if(n==0) return 1%m;
    int u = modpow(x,n/2,m);
    u = (u*u)%m;
    if(n%2==1) u = (u*x)%m;
    return u;
}
// modular exponentiation    --  O(log(n)) | space - O(1)
int modpow_bitmask(int x, int n, int m) {
    int res = 1;
    while(n>0) {
        if(n&1) res = (res*x)%m;
        x = (x*x)%m;
        n = n>>1;
    }
    return res;
}
int binMultiply(int x, int n, int m) {
    int res = 0;
    while(n>0) {
        if(n&1) res = (res+x)%m;
        x = (x+x)%m;
        n = n>>1;
    }
    return res;
}


// Numbers of factors  --  O(sqrt(n))
int no_of_factors(int n) {
    vi v = prime_factors(n);
    unordered_map<int,int> m;
    f(i,0,v.size()) m[v[i]]++;
    int res=1;
    for(auto x: m) res*=(x.ss+1);
    return res;
}
// Sum of fators    --      O(sqrt(n))
int sum_of_facors(int n) {
    vi v = prime_factors(n); 
    vi a; map<int,int> m; 
    f(i,0,v.size()) m[v[i]]++; 
    for(auto x: m) a.pb(x.ss+1);
    set<int> p(all(v));
    int res = 1; int i=0;
    for(int x: p) {
        res*=(pow(x,a[i])-1)/(x-1); i++;
    }
    return res;
}




/* <---------------------Combinatorics--------------------> */

// Derangements permutation
int solve(int n) { 
    if(n<=1) return 0;
    if(n==2) return 1;
    if(dp[n]) return dp[n];
    return dp[n]=(n-1)*(solve(n-2)+solve(n-1));   
}
// Iterative Approach  -  Time - O(n) | Space - O(n)
int solve(int n) { 
    dp[1]=0; dp[2]=1;
    for(int i=3; i<=n; ++i) {
        dp[i]=(i-1)*(dp[i-1]+dp[i-2]);
    }
    return dp[n];
}
// Iterative Approach  -  Time - O(n) | Space - O(1)
int solve(int n) { 
    int a=0; int b=1;
    for(int i=3; i<=n; ++i) {
        int cur=(i-1)*(a+b);
        a=b; b=cur;
    }
    return b;
}

int c2(int n){
    return (n*(n-1))/2;
}
// NCR
int ncr(int n, int r) {
    if(r > n-r) r = n-r;
    int C[r+1]; fill(C, C+r+1, 0); C[0] = 1;
    f(i,1,n+1) {
        fr(j,min(i,r)+1,1) C[j] = (C[j]+C[j-1]);
    }
    return C[r];
}







/* <-------------------String Algorithms------------------> */

// Patterm Matching    --   O(n*m)
bool isMatching(string s, string t) {
    int n = s.size();
    int m = t.size();
    f(i,0,(n-m)+1) {
        bool isFound = true;
        f(j,0,m) {
            if(s[i+j]!=t[j]) {
                isFound=false; 
                break;
            }
        }
        if(isFound) return true;
    }
    return false;
}
// KMP Algorithm   --   O(n+m)
vi getLps(string pattern) {
    int n = pattern.size();
    vi lps(n); // longest prefix which is also a sufix
    int i=1, j=0;
    while(i<n) {
        if(pattern[i]==pattern[j]) {
            lps[i]=j+1;
            i++, j++;
        }
        else {
            if(j!=0) j=lps[j-1];
            else lps[i]=0, i++;
        }
    }
    return lps;
}
bool kmpSearch(string text, string pattern) {
    int lenText = text.length();
    int lenPat = pattern.length();
    int i=0, j=0;
    vi lps = getLps(pattern);
    while(i<lenText && j<lenPat) {
        if(text[i]==pattern[j]) i++,j++;
        else {
            if(j!=0) j=lps[j-1];
            else i++;
        }
    }
    if(j==lenPat) return true;
    return false;
}
// Z - Algorithms        --       O(n+m) 
void buildZ(vi &Z, string str) {
    int l=0, r=0;
    int n = str.length();
    f(i,1,n) {
        if(i>r) { l=i; r=i;
            while(r<n && str[r-l]==str[r]) ++r;
            Z[i]=r-l; --r;
        }
        else {
            int k = i-l;
            if(Z[k]<=r-i) Z[i]=Z[k];
            else { l=i;
                while(r<n && str[r-l]==str[r]) ++r;
                Z[i]=r-l; --r;
            }
        }
    }
}
void searchString(string text, string pattern) {
    string str = pattern+"$"+text;
    int n = str.length();
    vi Z(n); buildZ(Z,str);
    f(i,0,n) {
        if(Z[i]==pattern.length()) {
            cout<<i-pattern.length()-1<<endl;
        }
    }
}
// calculate hashvalue      --     O(n)
int compute_hash(string const& s) {
    const int p = 31;
    const int m = 1e9 + 9;
    int hash_value = 0;
    int p_pow = 1;
    for(char c: s) {
        hash_value = (hash_value+(c-'a'+1)*p_pow)%m;
        p_pow = (p_pow*p)%m;
    }
    return hash_value;
}


// Longest Palindromic Substring   --   O(n^2)
int lps(string s) {
    int n = s.length();
    int max=0;
    f(i,0,n) {
        int l=i, r=i;
        // Odd length
        while(l>=0 && r<n && s[l]==s[r]) {
            int curr_l=r-l+1;
            if(curr_l>max) max=curr_l;
            --l; ++r;
        }
        // Even length
        l=i; r=i+1;
        while(l>=0 && r<n && s[i]==s[r]) {
            int curr_l=r-l+1;
            if(curr_l>max) max=curr_l;
            --l; ++r;
        }
    }
    return max;
}

// prefix max length in string  trei-DS
class trieNode {
public:
    trieNode** children;
    int weight;
    trieNode() {
        children = new trieNode*[26];
        f(i,0,26) children[i]=NULL;
        weight=0;
    }
};
void insert(string s, int weight, trieNode* root) {
    if(s.empty()) return;
    int idx = s[0]-'a';
    trieNode* child;
    if(root->children[idx]) child=root->children[idx];
    else {
        child = new trieNode();
        root->children[idx]=child;
    }
    if(child->weight < weight) child->weight=weight;
    insert(s.substr(1),weight,child);
}
int searchBest(string s, trieNode* root) {
    trieNode* current = root;
    int bestWeight=-1;
    f(i,0,s.length()) {
        int idx = s[i]-'a';
        trieNode* child = current->children[idx];
        if(child) {
            current=child;
            bestWeight=child->weight;
        }
        else return -1;
    }
    return bestWeight;
}







/* <--------------------Graph Algorithms------------------> */

// DFS
void dfs(vii &edges, vector<bool> &visited, int s) {
    if(visited[s]) return;
    visited[s] = true; cout<<s<<" ";
    for(auto u: edges[s]) dfs(edges,visited,u);
} // s->starting node
// BFS
void bfs(vii &edges, vector<bool>& visited, int s) {
    queue<int> q; q.push(s);
    while(!q.empty()) {
        int x = q.front(); cout<<x<<" ";
        for(auto u: edges[x]) { 
            if(!visited[u]) q.push(u),visited[u]=1; 
        }
        q.pop(); visited[x]=true;
    }
} // s->starting node
// traverse from all the nodes 
void bfs(vii &edges, vector<bool>& visited, int s, int n) {
    queue<int> q; q.push(s);
    for(int i=0; i<=n; ++i) {
        if(!visited[i]) { 
            if(!q.empty()) i--;
            else q.push(i);
            while(!q.empty()) {
                int x = q.front(); cout<<x<<" ";
                for(auto u: edges[x]) { 
                    if(!visited[u]) q.push(u),visited[u]=1; 
                }
                q.pop(); visited[x]=true;
            }
        }
    }
} // n->number of nodes, s->starting node

// has path
int dfs(vii edges,vector<bool> &visited, int s, int end) { cnt++;
    if(s==end) return 1;
    visited[s]=true;
    for(int u: edges[s]) {
        if(visited[u]==0 && dfs(edges,visited,u,end)==1) return 1; 
    }
    return 0;
}
// return all the vertices between the start and end 
int dfs(vii edges,vector<bool>&visited,int s,int end,vi& v) {
    if(s==end) return 1;
    visited[s]=true;
    for(int u: edges[s]) { 
        if(!visited[u]) {
            v.pb(u); if(dfs(edges,visited,u,end,v)) return 1;
        } 
    } 
    return 0;
}
// Dijkstra's Algorithm 
vi dijkstra_algo(vector<vector<pi>> edges, int n, int source) {
    vi dist(n+1,INT_MAX);
    dist[source]=0;
    set<pi> s; s.insert({0,source});

    while(!s.empty()) {
        pi x = *(s.begin());
        s.erase(x);
        for(auto it: edges[x.ss]) { 
            if(dist[it.ff] > dist[x.ss]+it.ss) {
                s.erase({dist[it.ff],it.ff});
                dist[it.ff]=dist[x.ss]+it.ss;
                s.insert({dist[it.ff],it.ff});    
            }
        }
    }
    return dist;
}






/* <---------------------Extras--------------------> */
// template<typename T>
// using oset = __gnu_pbds::tree<T,__gnu_pbds::null_type,less<T>,__gnu_pbds::rb_tree_tag,__gnu_pbds::tree_order_statistics_node_update>;




// trie implementation
class trieNode{
public: 
    trieNode *left;
    trieNode *right;
};

void insert(int n, trieNode* head) {
    trieNode *curr = head;
    for(int i=31; i>=0; --i) {
        int b = (n>>i)&1;
        if(b==0) {
            if(!curr->left) 
                curr->left = new trieNode();
            curr = curr->left;
        }
        else{
            if(!curr->right) 
                curr->right = new trieNode();
            curr = curr->right;
        }
    }
}

int findMaxXorPair(trieNode* head, vi v, int n) {
    int max_xor = INT_MIN;
    for(int i=0; i<n; ++i) {
        int value = v[i];
        trieNode* curr = head;
        int curr_xor = 0;
        for(int j=31; j>=0; --j) {
            int b = (value>>j)&1;
            if(b==0) {
                if(curr->right) {
                    curr_xor += pow(2,j);
                    curr = curr->right;
                }
                else curr = curr->left;
            }
            else {
                if(curr->left) {
                    curr_xor += pow(2,j);
                    curr = curr->left;
                }
                else curr = curr->right;
            }
        }
        max_xor = max(max_xor,curr_xor);
    }
    return max_xor;
}

// minimum path sum     -    google
int solve(vii v) {
    int n = v.size();
    int m = v[0].size();
    int ans =INT_MAX;
    f(i,0,n) {
        int a = v[i][0]; vi s; s.pb(a);
        f(j,1,m) { int b=1e3; int ele=-1;
            f(z,0,n) {
                if(abs(a-v[z][j]) < b){
                    b=abs(a-v[z][j]);
                    ele = v[z][j];
                }
            }
            s.pb(ele);
        }
        output(s); cout<<endl;
        int x = abs(*min_element(all(s)) - *max_element(all(s)));
        ans = min(ans, x);
    }
    return ans;
}


// microsft
int solution(vector<int>& v) {
    int n = v.size();
    int ans=0;
    for(int i=0; i<n; ++i) { int sum=0; int x = v[i];
        for(int j=i+1; j<n; ++j) { 
            if(v[j]>=x) { sum++; x=v[j]; }
            else break;
        } 
        x = v[i];
        for(int j=i-1; j>=0; --j) {
            if(v[j]>=x) { sum++; x=v[j]; }
            else break;
        } 
        ans=max(ans,sum);
    }
    return ans+1;
}
// visa
int lastStoneWeight(vector<int> weights) {
    priority_queue<int> q;
    int n = weights.size();
    for(int i=0; i<n; ++i) {
        q.push(weights[i]);
    }
    while(q.size()>1) {
        int x = q.top(); q.pop();  
        int y = q.top(); q.pop();
        if(x!=y) q.push(abs(x-y)); 
    }
    if(!q.size()) return 0;
    return q.top();
}

//samsung
sort(all(v)); 
int ans=0;
while(0<n) {
    int x = lower_bound(all(v),k-v[0])-v.begin();
    if(x==n) x--;
    else if(x!=0 && v[x]!=k-v[0]) x--;
    if(x!=0) v.erase(v.begin()+x);
    v.erase(v.begin());
    n=v.size(); ans++;
}
// Schlumberger 
string txt, pat; cin>>txt>>pat;
map<char, int> m;
for(int i=0; i<pat.size(); ++i) m[pat[i]]++;
int st=0, j=0;
auto s = m; int cnt=0;
f(i,0,txt.size()) {
    if(s.find(txt[i])!=s.end()) {
        s[txt[i]]--; 
        if(s[txt[i]]<0) { j++;
            f(z,st,i) { 
                s[txt[z]]++; 
                j--; st=z+1;  
                if(txt[z]==txt[i]) break; 
            }
        }
        else j++;
    }
    else { st=i+1; j=0; s=m; }
    if(j==pat.size()) {
        cout<<st<<endl; s[txt[st]]++; st++; j--;
    }
}