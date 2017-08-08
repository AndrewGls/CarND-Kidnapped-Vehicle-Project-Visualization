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
#include <random>
#include <limits>
//#include <sstream>
#include <iterator>

#define NDEBUG
#include <assert.h>

#include "helper_functions.h"
#include "particle_filter.h"

using namespace std;

// If USE_RESAMPLING_WHELL defined, the resamling wheel is used otherwise
// the discrete_distribution is used for normalization of weights.

//#define USE_RESAMPLING_WHELL

#define VISUALIZE_Associations

// Ramdon generator is defined here because particle_filter.h must not be changed for submission!
namespace
{
	default_random_engine gen(13);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

#if defined(VISUALIZE_Associations)
	num_particles = 100;// 50; // min 4 particles per filter.
#else
	num_particles = 100; // min 4 particles per filter.
#endif

	particles.resize(num_particles);
	weights.resize(num_particles, 1.);

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		auto& pa = particles[i];
		pa.id = i;
		pa.x = dist_x(gen);
		pa.y = dist_y(gen);
		pa.theta = dist_theta(gen);
		pa.weight = 1.;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	const double epsilon = std::numeric_limits<double>::epsilon();

	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_yaw(0, std_pos[2]);

	for (int i = 0; i < num_particles; ++i)
	{
		Particle& pa = particles[i];

		if (FP_ZERO == fpclassify(yaw_rate)) {
			// zero yaw rate
			//assert(false);
			pa.x += velocity * delta_t * cos(pa.theta) + N_x(gen);
			pa.y += velocity * delta_t * sin(pa.theta) + N_y(gen);
			pa.theta += N_yaw(gen);
		}
		else {
			// non-zero yaw rate
			const auto curr_theta = pa.theta;
			pa.theta += yaw_rate * delta_t;
			pa.x += velocity * (sin(pa.theta  ) - sin(curr_theta)) / yaw_rate;
			pa.y += velocity * (cos(curr_theta) - cos(pa.theta  )) / yaw_rate;

			pa.x += N_x(gen);
			pa.y += N_y(gen);
			pa.theta += N_yaw(gen);
		}

		// normalize theta in range [-2pi,+2pi]
		pa.theta = std::fmod(pa.theta, 2.*M_PI);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


}

namespace
{
	// Transfroms observations from Car-Space to Map-Space relative to particle orientation.
	inline void TransfromObservationsToMapSpace (const Particle& particle,
		const std::vector<LandmarkObs>& observations, std::vector<LandmarkObs>& t_observations)
	{
		// |x'| |cos(a) -sin(a) tx| |x|
		// |y'|=|sin(a)  cos(a) ty|*|y|
		// |1 | | 0        0     0| |1|

		for (int i = 0; i < observations.size(); i++)
		{
			const auto& obs = observations[i];
			auto& t_obs = t_observations[i];
			t_obs.id = obs.id;
			t_obs.x = cos(particle.theta) * obs.x - sin(particle.theta) * obs.y + particle.x;
			t_obs.y = sin(particle.theta) * obs.x + cos(particle.theta) * obs.y + particle.y;
		}
	}

	inline double MultVariateNormalDistrib(const LandmarkObs& predicted, const LandmarkObs& observated, const double std_landmark[])
	{
		// mean is predicted measurement.
		auto std_x = std_landmark[0];
		auto std_y = std_landmark[1];
		assert(FP_ZERO != fpclassify(std_x) && FP_ZERO != fpclassify(std_y));
		auto dx = observated.x - predicted.x;
		auto dy = observated.y - predicted.y;
		return exp(-dx*dx / (2 * std_x * std_x) - dy*dy / (2 * std_y * std_y)) / (2 * M_PI * std_x * std_y);
	}

	// The function returns the log of probability density function using a multivariate normal distribution function.
	inline double LogMultVariateNormalDistrib(const LandmarkObs& predicted, const LandmarkObs& observated, const double std_landmark[])
	{
		// Note: sqrt in scale factor in ln() is skipped for speedup. Calculated weights are normalized latter.
		// ln(p(x,y)) = -0.5 * (2*ln(2pi * std_x * std_y) + (x - mean_x)^2/std_x^2 + (y - mean_y)^2/std_y^2 )

		// mean is predicted measurement.
		auto std_x = std_landmark[0];
		auto std_y = std_landmark[1];
		assert(FP_ZERO != fpclassify(std_x) && FP_ZERO != fpclassify(std_y));

		const double coeff = 2. * log(2. * M_PI * std_x * std_y);
		auto dx = observated.x - predicted.x;
		auto dy = observated.y - predicted.y;
		return -0.5 * (coeff + dx*dx / (std_x*std_x) + dy*dy / (std_y*std_y));
	}

	inline double CalcMeasurementProbability(const std::vector<LandmarkObs>& predicted,
		const std::vector<LandmarkObs>& observations, const double std_landmark[])
	{
		// Calculates how likely a measurement should be.
		double prob = 1.;
		for (int k = 0; k < predicted.size(); k++) {
			prob *= MultVariateNormalDistrib(predicted[k], observations[k], std_landmark);
		}
		return prob;
	}

	// The function calculates the probability using log of probability density function to
	// avoid floating point underflow for very small values of probability.
	inline double CalcMeasurementProbabilityWithoutUnderflow(const std::vector<LandmarkObs>& predicted,
		const std::vector<LandmarkObs>& observations, const double std_landmark[])
	{
		// Calculates how likely a measurement should be.
		double log_prob = 0.;
		for (int k = 0; k < predicted.size(); k++) {
			log_prob += LogMultVariateNormalDistrib(predicted[k], observations[k], std_landmark);
		}
		return exp(log_prob);
	}

	void FindAssociation(std::vector<LandmarkObs>& predicted, const std::vector<LandmarkObs>& observations, const Map& map_landmarks)
	{
		assert(observations.size());
		predicted.resize(observations.size());

		std::vector<double> min_dist(observations.size(), INT_MAX);
		const auto& landmarks = map_landmarks.landmark_list;

		for (int i = 0; i < landmarks.size(); i++)
		{
			const auto& lm = landmarks[i];

			for (int j = 0; j < observations.size(); j++)
			{
				// find nearest landmark to observation
				const auto& obs = observations[j];
				auto distance = dist(lm.x_f, lm.y_f, obs.x, obs.y);
				if (distance < min_dist[j])
				{
					min_dist[j] = distance;

					auto& pred = predicted[j];
					pred.id = lm.id_i;
					pred.x = lm.x_f;
					pred.y = lm.y_f;
				}
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
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	assert(observations.size());
	if (!observations.size())
		return;

	std::vector<LandmarkObs> t_observations(observations.size());
	std::vector<LandmarkObs> predicted(observations.size());

	for (int i = 0; i < num_particles; ++i)
	{
		auto& pa = particles[i];

		// transform observations from car space to map space using particle position and orientation.
		TransfromObservationsToMapSpace(pa, observations, t_observations);

		// find nearest landmarks to observations
		FindAssociation(predicted, t_observations, map_landmarks);
		assert(predicted.size());

		// for visualization
#if defined(VISUALIZE_Associations)
		{
			pa.associations.resize(predicted.size());
			pa.sense_x.resize(predicted.size());
			pa.sense_y.resize(predicted.size());
			for (int i = 0; i < predicted.size(); ++i) {
				const auto& lm_obj = predicted[i];
				pa.associations[i] = lm_obj.id;
				pa.sense_x[i] = lm_obj.x;
				pa.sense_y[i] = lm_obj.y;
			}
		}
#endif

		// update weight of particle using calculated Multivariate-Gaussian probability
//		pa.weight = CalcMeasurementProbability(predicted, t_observations, std_landmark);
		pa.weight = CalcMeasurementProbabilityWithoutUnderflow(predicted, t_observations, std_landmark);
		weights[i] = pa.weight;
	}
}

#if !defined(USE_RESAMPLING_WHELL)

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// discrete_distribution does normalization of weights to transform weights into probability.
	discrete_distribution<int> d(weights.begin(), weights.end());

	std::vector<Particle> new_particles(num_particles);

	for (int i = 0; i < num_particles; i++) {
		new_particles[i] = particles[d(gen)];
	}

	particles.swap(new_particles);
}

#else

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// get max weight
	double max_weight = 0.;
	for (auto it = particles.cbegin(); it != particles.cend(); ++it) {
		max_weight = max(max_weight, it->weight);
	}

	// resampling wheel does normalization of weights to transform weights into probability.
	std::uniform_int_distribution<int> ud_start(0, num_particles - 1);
	std::uniform_real_distribution<double> ud_spin(0, 2. * max_weight);

	std::vector<Particle> new_particles(num_particles);

	// do resampling wheel
	int index = ud_start(gen);
	double beta = 0.;

	for (int i = 0; i < num_particles; i++)
	{
		beta += ud_spin(gen);

		const auto* pa = &particles[index];
		while (beta > pa->weight) {
			beta -= pa->weight;
			index = ++index % num_particles;
			pa = &particles[index];
		}

		new_particles[i] = *pa;
	}

	particles.swap(new_particles);
}

#endif

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
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

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}