#include <bits/stdc++.h>
using namespace std;
 
int main() {
 int n;
  cin >> n;
  
  vector<int> a(n);
  for(int i=0; i<n; i++){
  cin >> a[i];
  }
  
  set<int> s;
  
  for(int j=0; j<n; j++){
    s.insert(a[j]);
  }
  
  if(s.size() == n){
    cout << "YES" << endl;
  }
  else{
    cout << "NO" << endl;
  }
}
