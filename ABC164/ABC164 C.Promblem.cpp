#include <bits/stdc++.h>
using namespace std;
 
int main() {
  int n;
  cin >> n;
  
  vector<string> s(n);
  for (int i = 0; i < n; i++) {
    cin >> s[i];
  }
 
  set<string> num;
  for(int j=0; j<n; j++){
    num.insert(s[j]);
  }
  
  cout << num.size() << endl;
  
  
  
}
