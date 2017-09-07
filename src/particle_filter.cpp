/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

const double START_WEIGHT = 1.0;

random_device rd;
std::mt19937 gen(rd());


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 10;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  weights.clear();

  for(int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = START_WEIGHT;

    particles.push_back(particle);
    weights.push_back(START_WEIGHT);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);

  for(Particle& particle: particles) {
    double initial_yaw = particle.theta;
    double noise_x = dist_x(gen);
    double noise_y = dist_y(gen);
    double noise_theta = dist_theta(gen);

    if (abs(yaw_rate) > 0.001) {
      particle.x += (velocity / yaw_rate) * (sin(initial_yaw + (yaw_rate * delta_t)) - sin(initial_yaw)) + noise_x;
      particle.y += (velocity / yaw_rate) * (cos(initial_yaw) - cos(initial_yaw + (yaw_rate * delta_t))) + noise_y;
      particle.theta += (yaw_rate * delta_t) + noise_theta;
    } else {
      particle.x += ((velocity * delta_t) * cos(initial_yaw)) + noise_x;
      particle.y += ((velocity * delta_t) * sin(initial_yaw)) + noise_y;
      particle.theta += noise_theta;
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  for(LandmarkObs& observation: observations) {
    double shortest_distance = 1.0e6;

    for (LandmarkObs& prediction: predicted) {
      double current_dist = dist(observation.x, observation.y, prediction.x, prediction.y);

      if (current_dist < shortest_distance) {
        observation.id = prediction.id;
        shortest_distance = current_dist;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  for (int i = 0; i < particles.size(); i++) {
    std::vector<LandmarkObs> observations_transformed;
    std::vector<LandmarkObs> nearest_landmarks;
    Particle& particle = particles[i];

    // Find nearby landmarks
    for (Map::single_landmark_s& landmark: map_landmarks.landmark_list) {
      double distance = dist(particle.x, particle.y, landmark.x_f, landmark.y_f);
      if (distance < sensor_range) {
        LandmarkObs nearby_landmark;
        nearby_landmark.id = landmark.id_i;
        nearby_landmark.x = landmark.x_f;
        nearby_landmark.y = landmark.y_f;
        nearest_landmarks.push_back(nearby_landmark);
      }
    }

    // Transform car observation coordinates to map coordinates
    for (LandmarkObs& observation: observations) {
      LandmarkObs transformed;
      transformed.id = observation.id;
      transformed.x =  particle.x + (observation.x * cos(particle.theta)) - (sin(particle.theta) * observation.y);
      transformed.y =  particle.y + (observation.x * sin(particle.theta)) + (cos(particle.theta) * observation.y);

      observations_transformed.push_back(transformed);
    }

    // Assign nearest landmark ID to transformed observation
    dataAssociation(nearest_landmarks, observations_transformed);

    // Update particle's and particle filter's weight[s]
    double weight = START_WEIGHT;
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    for (LandmarkObs& observation: observations_transformed) {
      // observation.id's are associated with mapped landmarks ID's which are 1-index based.
      // We need to subtract by 1 when accessing corresponding landmark from map_landmarks.landmark_list.
      Map::single_landmark_s& landmark = map_landmarks.landmark_list[observation.id - 1];
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double a = pow(observation.x - landmark.x_f, 2) / (2.0 * pow(std_x, 2));
      double b = pow(observation.y - landmark.y_f, 2) / (2.0 * pow(std_y, 2));

      // Unlikely particles will result in low weight values, prompting them to drop off in resample() function through
      // discrete_distribution function
      weight *= exp(-(a + b)) / (2.0 * M_PI * std_x * std_y);

      associations.push_back(observation.id);
      sense_x.push_back(observation.x);
      sense_y.push_back(observation.y);
    }

    particle.weight = weight;
    weights[i] = weight;

    particle = SetAssociations(particle, associations, sense_x, sense_y);
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  discrete_distribution<int> resamplyzer(weights.begin(), weights.end());
  vector<Particle> resampled_particles;

  for(int i = 0; i < num_particles; i++) {
    int idx = resamplyzer(gen);
    resampled_particles.push_back(particles[idx]);
  }

  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
