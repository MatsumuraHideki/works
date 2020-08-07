#include <bits/stdc++.h>
using namespace std;
 
int main() {
  int n, m;
  cin >> n >> m;
  
  vector<int> a(n);
  for(int i=0; i<n; i++){
    cin >> a.at(i);
  }
  
  double allv=0;
  for(int j=0; j<n; j++){
    allv = allv + a.at(j);
  }
  
  int sum=0;
  for(int k=0; k<n; k++){
    if(a.at(k) >= allv/4/m){
      sum++;
    }
  }
  
  if(sum >= m){
    cout << "Yes" << endl;
  }
  else{
    cout << "No" << endl;
  }
  
  
   
}
