/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.
    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/

// this example shows how to use tiny-dnn library to fit data, by learning a
// sinus function.

// please also see:
// https://github.com/tiny-dnn/tiny-dnn/blob/master/docs/how_tos/How-Tos.md

#include <iostream>

#include "INTEGRITY.h"

#include "tiny_dnn/tiny_dnn.h"

int main()
{
  // create a simple network with 2 layer of 10 neurons each
  // input is x, output is sin(x)
  tiny_dnn::network<tiny_dnn::sequential> net;
  net << tiny_dnn::fully_connected_layer(1, 10);
  net << tiny_dnn::tanh_layer();
  net << tiny_dnn::fully_connected_layer(10, 10);
  net << tiny_dnn::tanh_layer();
  net << tiny_dnn::fully_connected_layer(10, 1);

  // create input and desired output on a period
  std::vector<tiny_dnn::vec_t> X;
  std::vector<tiny_dnn::vec_t> sinX;
 
  for (double x = -M_PI; x <= M_PI; x += 0.1)
  {
    tiny_dnn::vec_t vx    = {x};
    tiny_dnn::vec_t vsinx = {sin(x)};

    X.push_back(vx);
    sinX.push_back(vsinx);
  }

  // set learning parameters
  size_t batch_size = 16;    // 16 samples for each network weight update
  int epochs        = 2000;  // 2000 presentation of all samples
  tiny_dnn::adamax opt;

  // this lambda function will be called after each epoch
  int iEpoch              = 0;
  auto on_enumerate_epoch = [&]()
  {
    // compute loss and disp 1/100 of the time
    iEpoch++;
    if (iEpoch % 100) return;

    double loss = net.get_loss<tiny_dnn::mse>(X, sinX);
    std::cout << "epoch=" << iEpoch << "/" << epochs << " loss=" << loss
              << std::endl;
  };

  // learn
  std::cout << "learning the sinus function with 2000 epochs:" << std::endl;
  net.fit<tiny_dnn::mse>(opt, X, sinX, batch_size, epochs, []() {},
                         on_enumerate_epoch);

  std::cout << std::endl
            << "Training finished, now computing prediction results:"
            << std::endl;

  // compare prediction and desired output
  double fMaxError = 0.0;
 
  for (double x = -M_PI; x <= M_PI; x += 0.1)
  {
    tiny_dnn::vec_t xv = {x};
    double fPredicted   = net.predict(xv)[0];
    double fDesired     = sin(x);

    std::cout << "x=" << x << " sinX=" << fDesired
              << " predicted=" << fPredicted << std::endl;

    // update max error
    double fError = abs(fPredicted - fDesired);

    if (fMaxError < fError) fMaxError = fError;
  }

  std::cout << std::endl << "max_error=" << fMaxError << std::endl;

  return 0;
}
