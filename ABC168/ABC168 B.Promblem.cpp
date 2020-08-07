#include <bits/stdc++.h>
using namespace std;

int main() {
  int k;
  string s;
  cin >> k >> s;
  
  int u = s.size();
  
  if(u <= k){
    cout << s << endl;
  }
  
  else {
    for(int i=0; i<k; i++){
      if(i!=k-1){
      cout << s.at(i);
      }
      else {
        cout << s.at(i) << "..." << endl;
      }
    }
  }
}
