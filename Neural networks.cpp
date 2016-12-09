#include <iostream>
#include <stdexcept>
#include <cmath>
#include <stdlib.h>
#include <iomanip>
#include <time.h>

using namespace std;

typedef long long ll;

const double e = 2.71828182845904523536028747135266249775724709369995;
struct mat{ // Matrix for at most 10x10
    ll m, n;
    double M[10][10] = {};
    mat operator +(const mat &A){
      if(n != A.n || m != A.m)
      {
        throw invalid_argument( "Addition failed due to wrong dimensions" );
      }
      else
      {
        mat sol;
        for(ll i = 0; i < m; i++){
          for(ll j = 0; j < n; j++){
            sol.M[i][j] = M[i][j] + A.M[i][j];
          }
        }
        sol.m = m;
        sol.n = n;
        return sol;
      }
    }
    mat operator *(const mat &A){
      if(n != A.m){
        throw invalid_argument( "Multiplication failed due to incompatible matrices" );
      }
      mat sol;
      sol.m = m;
      sol.n = A.n;
      for(ll i = 0; i < m; i++){
        for(ll j = 0; j < A.n; j++){
          double sum = 0;
          for(ll i2 = 0; i2 < n; i2++){
     //       cout << "I " << i << " j " << j << " i2 "<< i2 << endl;
            sum += M[i][i2] * A.M[i2][j];
          }
          sol.M[i][j] = sum;
        }
      }
      return sol;
    }
};
mat T(mat A){
    mat sol;
    sol.m = A.n;
    sol.n = A.m;

    for(ll i = 0;i < A.m; i++){
      for(ll j = 0; j < A.n; j++){
        sol.M[j][i] = A.M[i][j];
      }
    }
    return sol;
}
void out(mat A){
    for(ll i = 0; i < A.m; i++){
      for(ll j = 0; j < A.n; j++){
        cout << A.M[i][j] << " ";
      }
      cout << endl;
    }
    cout << endl;
}

double fRand(double fMin, double fMax)
{

    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void genMat(mat & A){
    for(ll i = 0; i < A.m; i++){
      for(ll j = 0; j < A.n; j++){
        A.M[i][j] = fRand(-1,1);
    //    cout << A.M[i][j] << endl;
      }
    }
}

ll inputSize = 2;
ll outputSize = 1;
ll hiddenLayer = 3;
ll dataM = 3;
ll dataN = 2;

mat X;
mat Y;

mat W[10];
mat Z[5];

double cost(mat yhat, mat y){
    double error = 0;
    for(ll i = 0; i < dataM; i++){
      error += (y.M[i][0] - yhat.M[i][0]) * (y.M[i][0] - yhat.M[i][0]);
    }
    return error/2;
}
double sigmoid(double z){
    return (double)1/(1 + pow(e, -z));
}
mat sigmoid(mat z){
    mat sol;
    sol.m = z.m; sol.n = z.n;
    for(ll i = 0; i < z.m; i++){
      for(ll j = 0; j < z.n; j++){
        sol.M[i][j] = (double)1/(1 + pow(e, -z.M[i][j]));
      }
    }
    return sol;
}
void init(){
    X.m = dataM;
    X.n = dataN;
    X.M[0][0] = 3; X.M[0][1] = 5;
    X.M[1][0] = 5; X.M[1][1] = 1;
    X.M[2][0] = 10; X.M[2][1] = 2;

    Y.m = dataM;
    Y.n = 1;
    Y.M[0][0] = 0.75;
    Y.M[1][0] = 0.82;
    Y.M[2][0] = 0.93;


    W[1].m = inputSize;
    W[1].n = hiddenLayer;
    genMat(W[1]);
    W[2].m = hiddenLayer;
    W[2].n = outputSize;
    genMat(W[2]);
}
mat forward(mat X){
    Z[2] = X * W[1];
    Z[2] = sigmoid(Z[2]);
    Z[3] = Z[2] * W[2];
    return sigmoid(Z[3]);
}


int main()
{
    srand(time(NULL));
    double minim = 100;
    mat idx[3], sol;
    for(ll i = 0; i < 1000000; i++){
      init();
      mat yhat = forward(X);
      double error = cost(yhat, Y);
      if(error < minim){
        minim = error;
        idx[1] = W[1];
        idx[2] = W[2];
        sol = yhat;
      }
    }
    cout << "Minimum Error " << minim << endl;
    cout << "W[1] = \n";
    out(W[1]);
    cout << "W[2] = \n";
    out(W[2]);
    cout << "Sol  = \n";
    out(sol);
    return 0;
}
