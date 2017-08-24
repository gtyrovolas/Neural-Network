#include <iostream>
#include <stdexcept>
#include <cmath>
#include <stdlib.h>
#include <iomanip>
#include <time.h>
#include <assert.h>


using namespace std;

typedef long long ll;

const double e = 2.71828182845904523536028747135266249775724709369995;
// All these are the infastructure of a matrix
struct mat{ // Matrix for at most 10x10
    ll m, n; // matrix is m x n
    double M[10][10] = {};
    mat operator +(const mat &A){ // matrix addition
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
    mat operator *(const mat &A){ // matrix multiplication
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
    mat operator -(const mat &A){ // subtraction
      if(n != A.n || m != A.m)
      {
        throw invalid_argument( "Subtraction failed due to wrong dimensions" );
      }
      else
      {
        mat sol;
        for(ll i = 0; i < m; i++){
          for(ll j = 0; j < n; j++){
            sol.M[i][j] = M[i][j] - A.M[i][j];
          }
        }
        sol.m = m;
        sol.n = n;
        return sol;
      }
    }
    mat operator =(const mat &A){ // equals
      m = A.m;
      n = A.n;
      for(ll i = 0; i < m; i++){
        for(ll j = 0; j < n; j++){
          M[i][j] = A.M[i][j];
        }
      }
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
void out(mat A){ // output Matrix Contents
    for(ll i = 0; i < A.m; i++){
      for(ll j = 0; j < A.n; j++){
        cout << A.M[i][j] << " ";
      }
      cout << endl;
    }
    cout << endl;
}
void outD(mat A){ // ouput Dimensions
    cout << A.m << " " << A.n << endl;
}
mat multiply(mat A, mat B){ // scalar multiplication of two matrices
    if(A.m != B.m || A.n != B.n) throw invalid_argument("Scalar Multiplication of Matrices Failed\n");
    mat sol;
    sol.n = A.n;
    sol.m = A.m;
    for(ll i = 0; i < A.m; i++){
      for(ll j = 0; j < A.n; j++){
        sol.M[i][j] = A.M[i][j] * B.M[i][j];
      }
    }
    return sol;
}
double fRand(double fMin, double fMax) // generate random numbers
{

    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
void genMat(mat & A, double fMin, double fMax){ // fill a matrix with elements from -5 to 5
    for(ll i = 0; i < A.m; i++){
      for(ll j = 0; j < A.n; j++){
        A.M[i][j] = fRand(fMin,fMax);
      }
    }
}
mat scalMult(mat A, double sc){ // multiply a matrix A by a scalar
    for(ll i = 0; i < A.m; i++){
      for(ll j = 0; j < A.n; j++){
        A.M[i][j] *= sc;
      }
    }
    return A;
}




// setting HyperParameters
ll inputSize = 2; // number of input neurons
ll outputSize = 1; // number of output neurons
ll hiddenLayer = 3; // number of neurons in the hidden layers
ll dataM = 3;  // number of cases, also one dimension of the input matrix
ll dataN = 2;  // number of input neurons

mat X; // input
mat Y; // result

mat W[10]; // matrix that contains the weights of the synapses
mat Z[5];  // matrix that contains the raw version of the data
mat A[5]; // matrix that contains the activated data


double sigmoid(double z){ // sigmoid of a number
    return (double)1/(1 + pow(e, -z));
}
mat sigmoid(mat z){ // making the sigmoid of each item in a matrix
    mat sol;
    sol.m = z.m; sol.n = z.n;
    for(ll i = 0; i < z.m; i++){
      for(ll j = 0; j < z.n; j++){
        sol.M[i][j] = (double)1/(1 + pow(e, -(z.M[i][j])));
      }
    }
    return sol;
}

void init(){ // initialising the training
    X.m = dataM;
    X.n = dataN;
    X.M[0][0] = 3.0/10; X.M[0][1] = 10.0/10;
    X.M[1][0] = 5.0/10; X.M[1][1] = 2.0/10;
    X.M[2][0] = 10.0/10; X.M[2][1] = 4.0/10;
    Y.m = dataM;
    Y.n = 1;

    Y.M[0][0] = 0.75;
    Y.M[1][0] = 0.82;
    Y.M[2][0] = 0.93;


    W[1].m = inputSize;
    W[1].n = hiddenLayer;
    genMat(W[1], -2, 2);
    W[2].m = hiddenLayer;
    W[2].n = outputSize;
    genMat(W[2], -3, 3);
}

mat forward(mat X){  // forward propagation
    mat yHat;
    Z[2] = X * W[1];
    A[2] = sigmoid(Z[2]);
    Z[3] = A[2] * W[2];
    yHat = sigmoid(Z[3]);
    return yHat;
}
double cost(mat X, mat y){ // cost function
    mat yHat = forward(X);
    double error = 0;
    for(ll i = 0; i < dataM; i++){
      error += (y.M[i][0] - yHat.M[i][0]) * (y.M[i][0] - yHat.M[i][0]);
    }
    return error/2;
}
double cost2(mat A, mat B){ // cost function
    double error = 0;
    assert(A.m == B.m && A.n == B.n);
    for(ll i = 0; i < A.m; i++){
      error += (A.M[i][0] - B.M[i][0]) * (A.M[i][0] - B.M[i][0]);
    }
    return error/2;
}
double sigmoidPrime(double z){ // Returns the derivative of the sigmoid function
    double p = pow(e, -z);
    return p/((1 + p)*( 1+ p));
}

mat sigmoidPrime(mat z){ // returns the derivative of each element in a matrix
    mat sol;
    sol.n = z.n;
    sol.m = z.m;
    for(ll i = 0; i < sol.m; i++){
      for(ll j = 0; j < sol.n; j++){
        sol.M[i][j] = sigmoidPrime(z.M[i][j]);
      }
    }
    return sol;
}

void minErrorRand(){
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
 /*   cout << "Minimum Error " << minim << endl;
    cout << "W[1] = \n";
    out(W[1]);
    cout << "W[2] = \n";
    out(W[2]);
    cout << "Sol  = \n";
    out(sol);*/
}

void costFPrime(mat X, mat y, mat &yHat, mat &dJdW1, mat &dJdW2){
    yHat = forward(X);

    mat delta3 = multiply(yHat-y, sigmoidPrime(Z[3]));
    dJdW2 = T(A[2]) * delta3;
/*
    cout <<"Delta 3" << endl;
    out(delta3);
    cout << "W[2] Transposed\n";
    out(T(W[2]));
    cout << "Sigmoid Prime of Z[2] \n";
    out(sigmoidPrime(Z[2]));*/
    mat delta2 = multiply((delta3 * T(W[2])),sigmoidPrime(Z[2]));
    dJdW1 = T(X) * delta2;
}

void regulate(mat X, mat y, mat &yHat, double rate){
    mat dJdW1, dJdW2;
    costFPrime(X, y, yHat, dJdW1, dJdW2);
    W[1] = W[1] - scalMult(dJdW1, rate);
    W[2] = W[2] - scalMult(dJdW2, rate);
}

ll train(mat X, mat y, ll trials, double rate){
    mat dJdW1, dJdW2;
    mat best;
    ll cnt = 0;
    mat yHat;
    ll id;
    double mn = 100000;
    for(ll i = 0; i < trials; i++){
      regulate(X, y, yHat,rate);
   //   out(yHat);
      if(cost2(yHat, y) < mn){
        mn = cost2(yHat, y);
        id = i;
      }
    }
    return id;
   // cout << cost(yHat, y) << " after " << cnt << endl;
   // out(yHat);
}


// check with numerical gradient

void getGradients(mat &dJdW1, mat &dJdW2){ // function to get the Gradients computed by costFPrime
    mat yHat;
    costFPrime(X, Y, yHat, dJdW1, dJdW2);
}


void computeNumericalGradient(mat X, mat Y, mat &numdW1, mat &numdW2){ // Numerical gradient Descent
    mat dJdW1, dJdW2;
    getGradients(dJdW1, dJdW2);
    double eps = 0.0001;
    numdW1.n = dJdW1.n; numdW1.m = dJdW1.m;
    numdW2.n = dJdW2.n; numdW2.m = dJdW2.m;

    for(ll i = 0; i < numdW2.m; i++){
      for(ll j = 0; j < numdW2.n; j++){
        W[2].M[i][j] += eps;
        double loss2 = cost(X, Y);

        W[2].M[i][j] -= 2*eps;
        double loss1 = cost(X, Y);

        numdW2.M[i][j] = (loss2 - loss1)/ (2*eps);
        W[2].M[i][j] += eps;
      }
    }

    for(ll i = 0; i < numdW1.m; i++){
      for(ll j = 0; j < numdW1.n; j++){
        W[1].M[i][j] += eps;
        double loss2 = cost(X, Y);

        W[1].M[i][j] -= 2*eps;
        double loss1 = cost(X, Y);

        numdW1.M[i][j] = (loss2 - loss1)/ (2*eps);
        W[1].M[i][j] += eps;
      }
    }
}



int main(){
    srand(time(NULL));
    init();

    mat dJdW1, dJdW2;
    mat numdW1, numdW2;
    cout << train(X, Y, 10000, 1) << endl;
    cout << "yHat\n";
    out(forward(X));
    cout  << " vs Y" << endl;
    out(Y);
    cout << cost2(Y,forward(X)) << endl;

    cout << "Weights \n";
    out(W[1]);
    out(W[2]);


    mat X2;
    X2.n = 2;
    X2.m = 4;
    X2.M[0][0] = 0; X2.M[0][1] = 0;
    X2.M[1][0] = 0; X2.M[1][1] = 1;
    X2.M[2][0] = 1; X2.M[2][1] = 0;
    X2.M[3][0] = 1; X2.M[3][1] = 1;
    cout << "Results of X2" << endl;
    out(forward(X2));
}




/*
int main()
{
    srand(time(NULL));
    double minim = 100;
    mat B[3];
    mat bestY;
    ll id = 0;
    for(ll i = 0; i < 50; i++){
        cout << i << endl;
        init();
        train(X, Y, 10000);
        double c = cost(forward(X), Y);
        if(c < minim){
          minim = c;
          id = i;
          B[1] = W[1];
          B[2] = W[2];
          bestY = forward(X);
        }
    }
    out(B[1]);
    out(B[2]);
    out(bestY);
    cout << cost(bestY, Y) << endl;
    cout << id << endl;
    return 0;
}*/
