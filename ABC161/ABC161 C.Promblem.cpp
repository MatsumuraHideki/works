#include <bits/stdc++.h>
using namespace std;
 
int main() {
int64_t n, k;
  cin >> n >> k;
  
  int64_t d=n%k;
  
  if(k-d < d){
    cout << k-d << endl;
  }
  else{
    cout << d << endl;
  }
}
