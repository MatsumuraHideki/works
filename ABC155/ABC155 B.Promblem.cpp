#include <bits/stdc++.h>
using namespace std;
 
int main() {
	int n;
  	cin >> n;
  
  	vector<int> a(n);
  	for(int i=0; i<n; i++){
  	cin >> a[i];
  	}
  
  
  	int ok=0;
  
  for(int j=0; j<n; j++){
  if(a[j]%2==0 && a[j]%3 != 0 && a[j]%5 != 0){
    ok++;
    }
  }
  
  if(ok!=0){
    cout << "DENIED" << endl;
  }
  else{
    cout << "APPROVED" << endl;
  }
  
}
