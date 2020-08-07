#include <bits/stdc++.h>
using namespace std;
 
int main() {
  int n, k;
  cin >> n >> k;
  
  vector<int> p(n);
  for(int i=0; i<n; i++){
    cin >> p[i];
  }
 
  vector<double> exp(n);
  for(int j=0; j<k; j++){
  exp[0] += (p[j]+1)/2.0;
  }
  
  for(int m=0; m<n-k; m++){
    exp[m+1] = exp[m]+(p[m+k]-p[m])/2.0;
  }
  
  sort(exp.begin(), exp.end());
  
  cout << fixed << setprecision(15);
  
  cout << exp[n-1] << endl;
  
}
