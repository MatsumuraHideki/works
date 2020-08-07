#include <bits/stdc++.h>
using namespace std;
 
int main() {
  int64_t n, a, b;
  cin >> n >> a >> b;
  
  int64_t m = n /(a+b);
  
  if(n - ((a+b)*m) < a){
    cout << a*m + (n - ((a+b)*m)) << endl;
  }
  else{
    cout << a*(m+1) << endl;
  }
}
