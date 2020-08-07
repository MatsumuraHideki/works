#include <bits/stdc++.h>
using namespace std;
 
int main() {
int n, k;
  cin >> n >> k;
  
  int d=n/k;
  int nd=1;
  while(d!=0){
    nd++;
    d = d/k;
  }
  
  cout << nd << endl;
}
