#include <bits/stdc++.h>
using namespace std;
 
int main() {
  int x, n;
  cin >> x >> n;
  
  vector<int> p(n);
  for(int i=0; i<n; i++){
    cin >> p[i];
  }
  
  set<int> s;
  for(int k=0; k<n; k++){
    s.insert(p[k]);
  }
 
  int j=x;
  for(int m=1; m<101; m++){
    if(s.count(j)){
      if(m%2==0){
        j = j + m;
      }
      else{
        j = j - m;
      }
    }
    else{
      break;
    }
  }
  
  cout << j << endl;
}
 
