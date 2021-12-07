/* <---------------------CSES-------------------> */

// longest common substring uring DP   --   O(n^2)
int lcs(string a, string b) {
    int m = a.size(), n = b.size();
    int dp[m+1][n+1];
    int ans = 0;
    f(i, 0, m+1) {
        f(j, 0, n+1) {
            if (i==0 || j==0) dp[i][j] = 0;
            else if (a[i-1] == b[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
                ans = max(ans, dp[i][j]);
            }
            else dp[i][j] = 0;
        }
    }
    return ans;
}

// cses dp quesion Grid Paths  --  O(n^2)
int dp[1001][1001];
int grid_paths(vector<string> s) {
    memset(dp,0,sizeof(dp));
    int n = s.size();
    dp[0][0]=1; 
    f(i,0,n) {
        f(j,0,n) {
            if(i) dp[i][j]+=dp[i-1][j];
            if(j) dp[i][j]+=dp[i][j-1];
            dp[i][j]%=mod;
            if(s[i][j]=='*') dp[i][j]=0; 
        }
    }
    return dp[n-1][n-1];
}

// Interview question kartik arora DP
int dp[1000001][2];
int pseudo(int i, bool odd, int l, int h, int n) {ct++;
    if(i==n) {
        if(!odd) return 1;
        else return 0;
    } 
    if(dp[i][odd]!=-1) return dp[i][odd];
    int numOdds = (h-l+1)/2;

    if(l%2==1 && h%2==1) numOdds++;
    int numEvens = (h-l+1) - numOdds;
    return dp[i][odd] = (numOdds*pseudo(i+1,((odd)?0:1),l,h,n) + numEvens*pseudo(i+1,((odd)?1:0),l,h,n));
}// memset(dp,-1,sizeof(dp)); cout<<pseudo(0,0,l,h,n) 


// longest palindorme substring   --   O(n^3)
string solve(string s) {
    int n = s.length();
    int best_len = 0;
    string best_s = "";
    f(l,0,n) {
        f(r,l,n) {
            int len = r - l + 1;
            string subs = s.substr(l,len);
            if(is_palindrome(subs) && len>best_len) {
                best_len = len;
                best_s = subs;
            }
        }
    }
    return best_s;
}
// longest palindorme substring   --   O(n^2 log(n))
int good(int x, string s) {
    int n = s.length();
    for(int L = 0; L + x <= n; L++) {
        if(is_palindrome(s.substr(L, x))) {
            return L;
        }
    }
    return -1;
}
string solve(string s) {
    int best_len = 0;
    string best_s = "";
    int n = s.length();
    for(int parity : {0, 1}) {
        int low = 1, high = n;
        if(low % 2 != parity) low++;
        if(high % 2 != parity) high--;
        while(low <= high) {
            int mid = (low + high) / 2;
            if(mid % 2 != parity) mid++;
            if(mid > high) break;
            
            int tmp = good(mid, s);
            if(tmp != -1) {
                if(mid > best_len) {
                    best_len = mid;
                    best_s = s.substr(tmp, mid);
                }
                low = mid + 2;
            }
            else high = mid - 2;
        }
    }
    return best_s;
}
// longest palindorme substring   --   O(n^2)
string solve(string s) {
    int best_len = 0;
    string best_s = "";
    int n = s.length();
    for(int mid = 0; mid < n; mid++) {
        for(int x = 0; mid - x >= 0 && mid + x < n; x++) {
            if(s[mid-x] != s[mid+x]) break;
            int len = 2 * x + 1;
            if(len > best_len) {
                best_len = len;
                best_s = s.substr(mid - x, len);
            }
        }
    }
    for(int mid = 0; mid < n - 1; mid++) {
        for(int x = 1; mid - x + 1 >= 0 && mid + x < n; x++) {
            if(s[mid-x+1] != s[mid+x]) break;
            int len = 2 * x;
            if(len > best_len) {
                best_len = len;
                best_s = s.substr(mid - x + 1, len);
            }
        }
    }
    return best_s;
}

// Minimizing Coins DP  --   O(N*X)
int dp[MAX_SIZE]; int ct=0;
int solve(vi v, int x) { ct++;
    if(x==0) return 0;
    if(dp[x]) return dp[x];
    int ans = inf;
    for(auto s: v) {
        if(x>=s) ans=min(ans,solve(v,x-s)+1);
        else break;
    }
    dp[x] = ans;
    return ans;
}
// minimum path sum leetcode --  O(2^n)
int dp[100][100]; int ct=0;
int solve(vii v,int i,int j,int n,int m) { ct++;
    if(i==n-1 && j==m-1) return v[i][j];
    if(dp[i][j]) return dp[i][j];
    int ans=INT_MAX;
    if(j+1<m) ans = min(ans, solve(v,i,j+1,n,m)+v[i][j]);
    if(i+1<n) ans = min(ans, solve(v,i+1,j,n,m)+v[i][j]);
    dp[i][j] = ans; 
    // cout<<ans<<" "<<v[i][j]<<endl;
    return ans;
}
// minimum path sum  --  O(n^2)
int solve(vii v) {
    int n = v.size(); 
    int m = v[0].size();
    f(i,0,n+1) f(j,0,m+1) dp[i][j] = INT_MAX;
    dp[0][0]=v[0][0];
    f(i,0,n) {
        f(j,0,m) {
            if(j) dp[i][j] = min(dp[i][j], dp[i][j-1]+v[i][j]);
            if(i) dp[i][j] = min(dp[i][j], dp[i-1][j]+v[i][j]);
            // cout<<dp[i][j]<<" ";
        }
    }
    return dp[n-1][m-1];
}
// coin combination II -- O(n*x)
int solve(vi v, int x) {
    sort(all(v)); 
    dp[0]=1;
    for(int s: v) {
        f(i,1,x+1) {
            if(i>=s) dp[i] = (dp[i] + dp[i-s])%mod;
        }
    }   
    return dp[x];
}
// Unique Paths  --  ~ O(n*m)
int solve(int n, int m, int i, int j) { cnt++;
    if(i==n-1 && j==m-1) return 1;
    if(dp[i][j]) return dp[i][j];
    if(i+1<n) dp[i][j]+=solve(n,m,i+1,j);  
    if(j+1<m) dp[i][j]+=solve(n,m,i,j+1);  dp[i][j]%=mod; 
    return dp[i][j];
}
int solve(int n, int m) { cnt++;
    if(n==1 || m==1) return 1;
    if(dp[n][m]) return dp[n][m];
    return dp[n][m]=(solve(n-1,m)+solve(n,m-1))%mod;
}

//DP bitmask kartik arora 
int dp[21][1<<21];
int solve(vii& v, int i, int mask, int& n) {
    if(i==n) return 0;
    if(dp[i][mask] != -1) return dp[i][mask];
    
    int ans = INT_MAX;
    f(j,0,n) {
        if(mask&(1<<j)) 
            ans = min(ans, v[j][i]+solve(v,i+1,(mask^(1<<j)),n));
    }
    return dp[i][mask]=ans;
}// memset(dp,-1,sizeof(dp)); solve(v,0,(1<<n)-1,n)

// 0-1 Knapscak    using dp   --   O(n*w)
int solve(vi v, vi s, int w, int idx) { cnt++;
    if(w==0 || idx==v.size()) return 0;
    if(dp[idx][w]) return dp[idx][w];
    int ans = 0;
    f(i,idx,v.size()) {
        if(w>=s[i]) ans = max(ans, solve(v,s,w-s[i],i+1)+v[i]);
    }
    dp[idx][w]=ans;
    return ans;
} // v-> value, s->weight, idx=0, w=capacity

// Check paraenthsis  --  O(n)
bool isValid(string s) {
    int n = s.size();
    stack<char> st;
    f(i,0,n) {
        if(s[i]=='(') st.push(s[i]);
        if(s.empty()) return false;
        if(s[i]==')') st.pop();
    }
    return st.empty();
}
// Generate Parentheses   --   O()
vector<string>res;
void rev(string s, int open, int close) { cnt++; 
    if(open==0 && close==0) {
        res.pb(s);
        return;
    }
    if(open>0) rev(s+"(", open-1, close+1);
    if(close>0) rev(s+")", open, close-1);
}


// valid triangle   --  O(n^2 * log(n))
int triangleNumber(vector<int>& nums) {
    int n = nums.size();
    int ans=0;
    sort(nums.begin(),nums.end());
    for(int i=0; i<n-2; ++i) {
        for(int j=i+1; j<n-1; ++j) {
            int sum = nums[i]+nums[j];
            if(j+1<n) {
                int idx=lower_bound(nums.begin()+j+1,nums.end(),sum)-nums.begin();
                ans+=idx-(j+1);
            }
        }
    }
    return ans; 
}
// valid triangle  --  O(n^2)  or  O(2*n^2)
int triangleNumber(vector<int>& nums) {
    int n = nums.size();
    int ans=0;
    sort(nums.begin(),nums.end());
    for(int i=0; i<n-2; ++i) {
        int k = i+2;
        for(int j=i+1; j<n; ++j) {
            while(k<n && nums[i]+nums[j]>nums[k]) ++k;
            if(k>j) ans+=k-(j+1);
        }
    }
    return ans; 
}
// another approach  -  O(n^2)
int triangleNumber(vector<int>& nums) {
    int n = nums.size();
    int ans=0;
    sort(nums.begin(),nums.end());
    for(int k=n-1; k>=2; --k) {
        int i = 0;
        int j = k-1;
        while(i<j) {
            if(nums[i]+nums[j]>nums[k]) {
                ans+= j-i;
                j--;
            }
            else i++;
        }
    }
    return ans;
}

// leetcode
// jump game II  - bruteforce
int solve(vector<int> v, int i, int n) { cnt++;
    if(i==n-1) return 0;
    int ans=INT_MAX;
    for(int j=1; j<v[i]+1; ++j) {
        if(i+j<n) ans = min(ans,solve(v,i+j,n)+1);
        else break;
    }
    return ans;
}
// jump game II  -  using recurive dp
int solve(vector<int> v, int i, int n) { cnt++;
    if(i==n-1) return 0;
    if(dp[v[i]][i]) return dp[v[i]][i];
    dp[v[i]][i]=INT_MAX;
    for(int j=1; j<v[i]+1; ++j) {
        if(i+j<n) dp[v[i]][i] = min(dp[v[i]][i],solve(v,i+j,n)+1);
        else break;
    }
    return dp[v[i]][i];
}
// jump game II  -  iterative dp  
int solve(vi v, int n) {
    int ans=INT_MAX;
    fr(i,n-1,0) {
        dp[v[i]][i]=1e4+1; ans=INT_MAX;
        f(j,1,v[i]+1) {
            if(i+j<n)
                dp[v[i]][i] = min(dp[v[i]][i],dp[v[i+j]][i+j]+1);
            else break;
        } 
        ans=min(ans,dp[v[i]][i]);
    }
    return ans;
}

// Longest Consecutive sequence   --    O(n)
int solve(vi v) {
    int n=v.size();
    if(n<=1) return n;
    int ans=0; 
    unordered_map<int,int>m;
    for(int i=0; i<n; ++i) m[v[i]]++;
    for(int i=0; i<n; ++i) { 
        int count=1;
        if(!m[v[i]-1]) { cout<<v[i]<<" ";
            int x=v[i]+1;
            while(m[x++]) count++,cnt++;
        }
        ans=max(ans,count);
    }
    return ans;
}

// algoexpert - apparment hunting
int solve(vector<unordered_map<string, bool>> blocks) {
    int n=blocks.size();
    int ans=INT_MAX;
    for(int i=0; i<n; ++i) {
        bool gym=0,school=0,store=0;
        int l=i,r=i; bool flag=0;
        while(l>=0 || r<n) {
            if(l>=0) {
                if(gym==false) gym=blocks[l]["gym"];
                if(school==false) school=blocks[l]["school"];
                if(store==false) store=blocks[l]["store"];
                // cout<<gym<<" "<<school<<" "<<store<<endl;
                if(gym && school && store) { flag=1; break; }
            }
            if(r<n) {
                if(gym==false) gym=blocks[r]["gym"];
                if(school==false) school=blocks[r]["school"];
                if(store==false) store=blocks[r]["store"];
                // cout<<gym<<" "<<school<<" "<<store<<endl;
                if(gym && school && store) { flag=1; break; }
            }
            l--;
            r++;
        } 
        if(flag) ans=min(ans, r-i);
        cout<<"ans: "<<ans<<endl;
    }
    return ans;
}
// vector<unordered_map<string, bool>> blocks;
// blocks.pb({
//     {"gym",false},
//     {"school",true},
//     {"store",false}});
// blocks.pb({
//     {"gym",true},
//     {"school",false},
//     {"store",false}});
// blocks.pb({
//     {"gym",true},
//     {"school",true},
//     {"store",false}});
// blocks.pb({
//     {"gym",false},
//     {"school",true},
//     {"store",false}});
// blocks.pb({
//     {"gym",false},
//     {"school",true},
//     {"store",true}});