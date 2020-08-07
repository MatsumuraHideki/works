#include <bits/stdc++.h>
using namespace std;
 
int main() {
  int n, k;
  cin >> n >> k;
  
  vector<int> d(k);
  vector<vector<int>> a(k,vector<int>(n));
  
  for(int i=0; i<k; i++){
    cin >> d[i];
  for(int j=0; j<d[i]; j++){  
    cin >> a[i][j];
  }
  }
 
  set<int> s;
  for(int x=0; x<k; x++){
    for(int y=0; y<d[x]; y++){
      s.insert(a[x][y]);
    }
  }
  
  if(s.count(0)){
  cout << n-s.size()-1 << endl;
  }
  else{
    cout << n-s.size() << endl;
  }
}
