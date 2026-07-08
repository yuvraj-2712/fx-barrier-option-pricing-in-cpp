/*
    FX BARRIER OPTION PRICER USING MONTE CARLO SIMULATION
    -------------------------------------------------------
    1. I simulate many possible future paths of an FX exchange rate using
       Geometric Brownian Motion (GBM), discretized with the Euler-Maruyama method.
    2. For each simulated path, I then check if it breaches a "barrier" level.
       - Up-and-Out Call : option dies if price goes ABOVE barrier
       - Down-and-In Put  : option only becomes "alive" if price goes BELOW barrier
    3. I price the option three ways:
       a) Standard (Naive) Monte Carlo
       b) Antithetic Variates (a variance reduction trick)
       c) Control Variates (using the Black-Scholes vanilla price as a control)
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <string>
#include <map>
#include <sstream>
using namespace std;
// ------------------------------------------------------------------
// SECTION 0: INPUT FILE READER
// Reads "key=value" pairs from a text file (inputs.txt) that the
// Python script (fetch_market_data.py) generates from live market data.
// If the file is missing, then I fall back to sensible hardcoded defaults
// so the program still runs standalone.
// ------------------------------------------------------------------
map<string, double> readInputsFromFile(const string& filename) {
    map<string, double> values;
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "NOTE: " << filename << " not found. Using default hardcoded inputs.\n";
        return values; // empty map -- caller will use defaults
    }
    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        size_t equalPos = line.find('=');
        if (equalPos == string::npos) continue; // skip malformed lines
        string key = line.substr(0, equalPos);
        string valueStr = line.substr(equalPos + 1);
        double value = stod(valueStr);
        values[key] = value;
    }
    cout << "Loaded market-derived inputs from " << filename << "\n";
    return values;
}
// ------------------------------------------------------------------
// SECTION 0b: CLOSED-FORM VALIDATION (Reiner & Rubinstein, 1991)
// -----------------------------------------------------------------
// My Monte Carlo engine gives us an ESTIMATE with some random error.
// But how do I know the estimate is actually CORRECT (not just
// "consistent")? The answer is to compare it against a known, exact
// formula for the same option, when one exists.
//
// Reiner and Rubinstein (1991) derived a closed-form (exact) pricing
// formula for European barrier options under plain Black-Scholes/GBM
// dynamics. Implemented below for the specific case used in this
// project: an UP-AND-OUT CALL with barrier H above strike K (K < H,
// H > S0), which is the standard, most common case for this option type.
//
// If my Monte Carlo price (especially at large N) lands close to this
// closed-form number, that helps confirm my simulation engine is
// correctly built, not just "converging to some number".
// -----------------------------------------------------------------
double standaloneNormalCDF(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

double upAndOutCallClosedForm(double S0, double K, double H, double r, double sigma, double T) {
    if (S0 >= H) {
        return 0.0; // already knocked out at time zero
    }
    if (K >= H) {
        return 0.0; // different edge case, not used in this project
    }

    double sqrtT = sqrt(T);
    double mu = (r - 0.5 * sigma * sigma) / (sigma * sigma);
    double eta = -1.0; // eta = -1 for "up" barrier options, +1 for "down" barrier options

    double x1 = (log(S0 / K) / (sigma * sqrtT)) + (1 + mu) * sigma * sqrtT;
    double x2 = (log(S0 / H) / (sigma * sqrtT)) + (1 + mu) * sigma * sqrtT;
    double y1 = (log(H * H / (S0 * K)) / (sigma * sqrtT)) + (1 + mu) * sigma * sqrtT;
    double y2 = (log(H / S0) / (sigma * sqrtT)) + (1 + mu) * sigma * sqrtT;

    double termA = S0 * standaloneNormalCDF(x1) - K * exp(-r * T) * standaloneNormalCDF(x1 - sigma * sqrtT);
    double termB = S0 * standaloneNormalCDF(x2) - K * exp(-r * T) * standaloneNormalCDF(x2 - sigma * sqrtT);

    // NOTE: the (H/S0) power term uses TWO DIFFERENT exponents depending on
    // whether it multiplies the S0 part or the K part of the formula
    // (2*(mu+1) for the S0 part, 2*mu for the K part). Mixing these up is a
    // very easy mistake to make -- worth double-checking if you ever adapt this.
    double termC = S0 * pow(H / S0, 2 * (mu + 1)) * standaloneNormalCDF(eta * y1)
                  - K * exp(-r * T) * pow(H / S0, 2 * mu) * standaloneNormalCDF(eta * y1 - eta * sigma * sqrtT);
    double termD = S0 * pow(H / S0, 2 * (mu + 1)) * standaloneNormalCDF(eta * y2)
                  - K * exp(-r * T) * pow(H / S0, 2 * mu) * standaloneNormalCDF(eta * y2 - eta * sigma * sqrtT);

    double price = termA - termB + termC - termD;
    if (price < 0) price = 0.0;
    return price;
}

// ------------------------------------------------------------------
// SECTION 1: BARRIER OPTION CLASS

// This class simply stores all the details of the option contract.
// ------------------------------------------------------------------
class BarrierOption {
public:
    double S0;       // current FX spot rate
    double K;        // strike price
    double barrier;  // barrier level
    double r;        // risk-free interest rate
    double sigma;    // volatility
    double T;         // time to maturity (in years)
    string type;      // "up-and-out-call" or "down-and-in-put"
    // Constructor
    BarrierOption(double S0_, double K_, double barrier_, double r_,
                  double sigma_, double T_, string type_) {
        S0 = S0_;
        K = K_;
        barrier = barrier_;
        r = r_;
        sigma = sigma_;
        T = T_;
        type = type_;
    }
    // ------------------------------------------------------------------
    // CONTINUITY CORRECTION (Broadie, Glasserman, Kou - 1997)
    // Problem: my simulation only checks the barrier at a FINITE number
    // of discrete time steps (numSteps), but a real barrier option is
    // monitored CONTINUOUSLY. Checking only at discrete points means I
    // sometimes "miss" a barrier breach that happened briefly BETWEEN
    // two steps -- this makes discretely-monitored knock-out options
    // look too expensive (or knock-in options too cheap) compared to
    // their true continuous-monitoring price.
    //
    // The fix: nudge the barrier slightly AWAY from the spot price before
    // simulating. This well-known correction (beta = 0.5826) compensates
    // for the discretization bias without changing anything else about
    // the simulation loop.
    //
    // Formula:
    //   Upper barrier (up-and-out, up-and-in): H_adjusted = H * exp(+beta*sigma*sqrt(dt))
    //   Lower barrier (down-and-out, down-and-in): H_adjusted = H * exp(-beta*sigma*sqrt(dt))
    // ------------------------------------------------------------------
    void applyContinuityCorrection(int numSteps) {
        const double beta = 0.5826; // constant from Broadie-Glasserman-Kou (1997)
        double dt = T / numSteps;
        double adjustmentFactor = exp(beta * sigma * sqrt(dt));
        bool isUpperBarrier = (type == "up-and-out-call" || type == "up-and-in-call");
        if (isUpperBarrier) {
            barrier = barrier * adjustmentFactor;       // push upper barrier further up
        } else {
            barrier = barrier / adjustmentFactor;        // push lower barrier further down
        }
    }
    // This function looks at ONE simulated price path and decides
    // the payoff of the option based on barrier conditions.
    double getPayoff(const vector<double>& path) {
        int n = path.size();
        double finalPrice = path[n - 1];
        if (type == "up-and-out-call") {
            // If price ever crosses ABOVE barrier, option is knocked out (payoff = 0)
            for (int i = 0; i < n; i++) {
                if (path[i] >= barrier) {
                    return 0.0;
                }
            }
            // If never knocked out, behaves like a normal call option
            double payoff = finalPrice - K;
            if (payoff < 0) payoff = 0.0;
            return payoff;
        }
        else if (type == "down-and-in-put") {
            // Option only "activates" if price falls BELOW barrier at some point
            bool activated = false;
            for (int i = 0; i < n; i++) {
                if (path[i] <= barrier) {
                    activated = true;
                    break;
                }
            }
            if (!activated) {
                return 0.0; // never activated, so worthless
            }
            double payoff = K - finalPrice;
            if (payoff < 0) payoff = 0.0;
            return payoff;
        }
        // Default fallback (should not happen if type is correct)
        return 0.0;
    }
};
// ------------------------------------------------------------------
// SECTION 2: MONTE CARLO ENGINE CLASS
// This class handles path generation and the three pricing methods.
// ------------------------------------------------------------------
class MonteCarloEngine {
public:
    int numSimulations;   // how many random paths to simulate
    int numSteps;         // how many time-steps per path (for barrier checking)
    mt19937 rng;          // random number generator engine
    MonteCarloEngine(int numSimulations_, int numSteps_) {
        numSimulations = numSimulations_;
        numSteps = numSteps_;
        rng.seed(42); // fixed seed so results are reproducible
    }
    // Generates ONE path of the FX rate using Euler-Maruyama discretization
    // of Geometric Brownian Motion:
    //   dS = r*S*dt + sigma*S*dW
    // Discretized as:
    //   S(t+dt) = S(t) + r*S(t)*dt + sigma*S(t)*sqrt(dt)*Z
    // where Z is a standard normal random number.
    // "antitheticFlip" lets us reuse the SAME random numbers with a flipped
    // sign, this is the core trick behind Antithetic Variates.
    vector<double> generatePath(const BarrierOption& opt, vector<double>& storedZ,
                                 bool useStoredZ, bool antitheticFlip) {
        double dt = opt.T / numSteps;
        vector<double> path(numSteps + 1);
        path[0] = opt.S0;
        normal_distribution<double> normalDist(0.0, 1.0);
        for (int i = 1; i <= numSteps; i++) {
            double Z;
            if (useStoredZ) {
                // reuse previously generated random number (for antithetic pass)
                Z = storedZ[i - 1];
                if (antitheticFlip) Z = -Z;
            } else {
                Z = normalDist(rng);
                storedZ[i - 1] = Z; // save it in case I need the antithetic pair later
            }
            double prevPrice = path[i - 1];
            double drift = opt.r * prevPrice * dt;
            double diffusion = opt.sigma * prevPrice * sqrt(dt) * Z;
            path[i] = prevPrice + drift + diffusion;
            if (path[i] < 0) path[i] = 0.0; // FX rate cannot go negative
        }
        return path;
    }
    // ----------------------------------------------------------
    // METHOD A: STANDARD (NAIVE) MONTE CARLO
    // ----------------------------------------------------------
    void standardMonteCarlo(BarrierOption& opt, double& price, double& stdError,
                             double& timeTaken, vector<double>& samplePathForPlot) {
        auto startTime = chrono::high_resolution_clock::now();
        vector<double> payoffs(numSimulations);
        vector<double> dummyZ(numSteps);
        for (int i = 0; i < numSimulations; i++) {
            vector<double> path = generatePath(opt, dummyZ, false, false);
            double payoff = opt.getPayoff(path);
            payoffs[i] = payoff;
            if (i == 0) samplePathForPlot = path; // save one path just for visualization
        }
        // discount average payoff back to today using risk-free rate
        double sumPayoff = 0.0;
        for (int i = 0; i < numSimulations; i++) sumPayoff += payoffs[i];
        double meanPayoff = sumPayoff / numSimulations;
        price = exp(-opt.r * opt.T) * meanPayoff;
        // standard error tells us how "noisy" my estimate is
        double sumSquaredDiff = 0.0;
        for (int i = 0; i < numSimulations; i++) {
            double diff = payoffs[i] - meanPayoff;
            sumSquaredDiff += diff * diff;
        }
        double variance = sumSquaredDiff / (numSimulations - 1);
        double sampleStdDev = sqrt(variance);
        stdError = exp(-opt.r * opt.T) * sampleStdDev / sqrt((double)numSimulations);
        auto endTime = chrono::high_resolution_clock::now();
        timeTaken = chrono::duration<double>(endTime - startTime).count();
    }
    // ----------------------------------------------------------
    // METHOD B: ANTITHETIC VARIATES
    // Idea: for every random path generated with Z, ALSO generate a
    // "mirror" path using -Z. Averaging the two payoffs together reduces
    // variance because the two paths' errors partly cancel out.
    // ----------------------------------------------------------
    void antitheticMonteCarlo(BarrierOption& opt, double& price, double& stdError,
                               double& timeTaken) {
        auto startTime = chrono::high_resolution_clock::now();
        vector<double> combinedPayoffs(numSimulations);
        for (int i = 0; i < numSimulations; i++) {
            vector<double> storedZ(numSteps);
            // path 1: normal random draw (Z gets stored)
            vector<double> path1 = generatePath(opt, storedZ, false, false);
            double payoff1 = opt.getPayoff(path1);
            // path 2: mirror path using -Z (antithetic pair)
            vector<double> path2 = generatePath(opt, storedZ, true, true);
            double payoff2 = opt.getPayoff(path2);
            // average the pair -- this is the key variance reduction step
            combinedPayoffs[i] = (payoff1 + payoff2) / 2.0;
        }
        double sumPayoff = 0.0;
        for (int i = 0; i < numSimulations; i++) sumPayoff += combinedPayoffs[i];
        double meanPayoff = sumPayoff / numSimulations;
        price = exp(-opt.r * opt.T) * meanPayoff;
        double sumSquaredDiff = 0.0;
        for (int i = 0; i < numSimulations; i++) {
            double diff = combinedPayoffs[i] - meanPayoff;
            sumSquaredDiff += diff * diff;
        }
        double variance = sumSquaredDiff / (numSimulations - 1);
        double sampleStdDev = sqrt(variance);
        stdError = exp(-opt.r * opt.T) * sampleStdDev / sqrt((double)numSimulations);
        auto endTime = chrono::high_resolution_clock::now();
        timeTaken = chrono::duration<double>(endTime - startTime).count();
    }
    // Standard Normal CDF approximation, needed for the Black-Scholes formula below.
    // This uses the well-known error function relationship: N(x) = 0.5*(1+erf(x/sqrt(2)))
    double normalCDF(double x) {
        return 0.5 * (1.0 + erf(x / sqrt(2.0)));
    }
    // Closed-form Black-Scholes price for a VANILLA European call.
    // I use this as my "Control Variate" because I know its true price exactly,
    // so I can use the error in pricing THIS option to correct my barrier price.
    double blackScholesCall(double S0, double K, double r, double sigma, double T) {
        double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
        double d2 = d1 - sigma * sqrt(T);
        return S0 * normalCDF(d1) - K * exp(-r * T) * normalCDF(d2);
    }
    // ----------------------------------------------------------
    // METHOD C: CONTROL VARIATES
    // Idea: I price a VANILLA call alongside my barrier option on the SAME
    // simulated paths. I know the vanilla call's true price (Black-Scholes).
    // If my simulated vanilla price is off by some amount, my barrier price
    // is probably off by a similar amount (they use the same random numbers).
    // So I adjust: 
    //   AdjustedPrice = BarrierMCPrice - (VanillaMCPrice - TrueVanillaPrice)
    // ----------------------------------------------------------
    void controlVariateMonteCarlo(BarrierOption& opt, double& price, double& stdError,
                                   double& timeTaken) {
        auto startTime = chrono::high_resolution_clock::now();
        vector<double> barrierPayoffs(numSimulations);
        vector<double> vanillaPayoffs(numSimulations);
        vector<double> dummyZ(numSteps);
        for (int i = 0; i < numSimulations; i++) {
            vector<double> path = generatePath(opt, dummyZ, false, false);
            double bPayoff = opt.getPayoff(path);
            barrierPayoffs[i] = bPayoff;
            // vanilla call payoff using the SAME simulated path (same random numbers)
            double finalPrice = path[path.size() - 1];
            double vPayoff = finalPrice - opt.K;
            if (vPayoff < 0) vPayoff = 0.0;
            vanillaPayoffs[i] = vPayoff;
        }
        // compute means
        double sumBarrier = 0.0, sumVanilla = 0.0;
        for (int i = 0; i < numSimulations; i++) {
            sumBarrier += barrierPayoffs[i];
            sumVanilla += vanillaPayoffs[i];
        }
        double meanBarrier = sumBarrier / numSimulations;
        double meanVanilla = sumVanilla / numSimulations;
        // find optimal coefficient "beta" using covariance / variance formula
        // beta = Cov(barrier, vanilla) / Var(vanilla)
        double covariance = 0.0, varianceVanilla = 0.0;
        for (int i = 0; i < numSimulations; i++) {
            double diffB = barrierPayoffs[i] - meanBarrier;
            double diffV = vanillaPayoffs[i] - meanVanilla;
            covariance += diffB * diffV;
            varianceVanilla += diffV * diffV;
        }
        double beta = covariance / varianceVanilla;
        // true discounted Black-Scholes price of the vanilla call
        double trueVanillaPrice = blackScholesCall(opt.S0, opt.K, opt.r, opt.sigma, opt.T);
        double simulatedVanillaPrice = exp(-opt.r * opt.T) * meanVanilla;
        // adjust each barrier payoff using the control variate formula
        vector<double> adjustedPayoffs(numSimulations);
        for (int i = 0; i < numSimulations; i++) {
            adjustedPayoffs[i] = barrierPayoffs[i] - beta * (vanillaPayoffs[i] - trueVanillaPrice);
        }
        double sumAdjusted = 0.0;
        for (int i = 0; i < numSimulations; i++) sumAdjusted += adjustedPayoffs[i];
        double meanAdjusted = sumAdjusted / numSimulations;
        price = exp(-opt.r * opt.T) * meanAdjusted;
        double sumSquaredDiff = 0.0;
        for (int i = 0; i < numSimulations; i++) {
            double diff = adjustedPayoffs[i] - meanAdjusted;
            sumSquaredDiff += diff * diff;
        }
        double varianceAdj = sumSquaredDiff / (numSimulations - 1);
        double sampleStdDev = sqrt(varianceAdj);
        stdError = exp(-opt.r * opt.T) * sampleStdDev / sqrt((double)numSimulations);
        auto endTime = chrono::high_resolution_clock::now();
        timeTaken = chrono::duration<double>(endTime - startTime).count();
        // silence unused variable warning;
        (void)simulatedVanillaPrice;
        (void)trueVanillaPrice;
    }
};
// ------------------------------------------------------------------
// SECTION 3: MAIN FUNCTION
// This is where I set up the option, run all three pricing methods,
// and write results out to CSV files for Python to visualize.
// ------------------------------------------------------------------
int main() {
    // ---- Option parameters ----
    // Default hardcoded values (used only if inputs.txt is not found).
    double S0 = 83.0;        // current FX spot rate (e.g., USD/INR)
    double K = 84.0;         // strike
    double barrier = 88.0;   // knock-out barrier level
    double r = 0.05;         // risk-free rate (5%)
    double sigma = 0.10;     // volatility (10%)
    double T = 1.0;          // 1 year maturity
    string optionType = "up-and-out-call";
    // Try to load real market-derived inputs written by fetch_market_data.py.
    // If the file doesn't exist, the hardcoded defaults above are used instead.
    map<string, double> marketInputs = readInputsFromFile("inputs.txt");
    if (!marketInputs.empty()) {
        S0 = marketInputs["S0"];
        K = marketInputs["K"];
        barrier = marketInputs["barrier"];
        r = marketInputs["r"];
        sigma = marketInputs["sigma"];
        T = marketInputs["T"];
    }
    double originalBarrier = barrier; // keep a copy purely for printing/comparison
    BarrierOption option(S0, K, barrier, r, sigma, T, optionType);
    int numSteps = 252; // roughly daily monitoring over 1 year
    // Apply the Broadie-Glasserman-Kou continuity correction so my
    // discretely-monitored simulation better matches a continuously
    // monitored barrier option.
    option.applyContinuityCorrection(numSteps);
    cout << "\n----- OPTION SETUP -----\n";
    cout << "S0=" << S0 << "  K=" << K << "  r=" << r << "  sigma=" << sigma << "  T=" << T << "\n";
    cout << "Original barrier             : " << originalBarrier << "\n";
    cout << "Continuity-corrected barrier : " << option.barrier << "\n";
    cout << "-------------------------------\n\n";

    // ---- Closed-form validation (Reiner & Rubinstein, 1991) ----
    // Computed using the ORIGINAL barrier (continuous-monitoring assumption),
    // so it can be compared fairly against my discretely-monitored MC prices.
    double closedFormPrice = upAndOutCallClosedForm(S0, K, originalBarrier, r, sigma, T);
    cout << "----- CLOSED-FORM VALIDATION (Reiner & Rubinstein, 1991) -----\n";
    cout << "Closed-form Up-and-Out Call price : " << closedFormPrice << "\n";
    cout << "(computed using the ORIGINAL barrier, i.e. continuous monitoring assumption)\n";
    cout << "-------------------------------\n\n";

    ofstream closedFormFile("closed_form_price.csv");
    closedFormFile << "ClosedFormPrice\n";
    closedFormFile << closedFormPrice << "\n";
    closedFormFile.close();
    // I will test a few different simulation sizes to show convergence
    vector<int> simulationSizes = {10000, 50000, 100000, 500000, 1000000};
    ofstream resultsFile("results.csv");
    resultsFile << "NumSimulations,Method,Price,StdError,TimeSeconds\n";
    vector<double> samplePath; // will hold one path for plotting
    for (int n : simulationSizes) {
        MonteCarloEngine engine(n, numSteps);
        double price, stdError, timeTaken;
        // Method A: Standard Monte Carlo
        engine.standardMonteCarlo(option, price, stdError, timeTaken, samplePath);
        resultsFile << n << ",Standard," << price << "," << stdError << "," << timeTaken << "\n";
        cout << "[Standard]   N=" << n << "  Price=" << price
             << "  StdError=" << stdError << "  Time=" << timeTaken << "s\n";
        // Method B: Antithetic Variates
        engine.antitheticMonteCarlo(option, price, stdError, timeTaken);
        resultsFile << n << ",Antithetic," << price << "," << stdError << "," << timeTaken << "\n";
        cout << "[Antithetic] N=" << n << "  Price=" << price
             << "  StdError=" << stdError << "  Time=" << timeTaken << "s\n";
        // Method C: Control Variates
        engine.controlVariateMonteCarlo(option, price, stdError, timeTaken);
        resultsFile << n << ",ControlVariate," << price << "," << stdError << "," << timeTaken << "\n";
        cout << "[ControlVar] N=" << n << "  Price=" << price
             << "  StdError=" << stdError << "  Time=" << timeTaken << "s\n";
        cout << "-----------------------------------------------------\n";
    }
    resultsFile.close();
    // Save one sample path so Python can plot what a simulated FX path looks like
    ofstream pathFile("sample_path.csv");
    pathFile << "Step,Price\n";
    for (size_t i = 0; i < samplePath.size(); i++) {
        pathFile << i << "," << samplePath[i] << "\n";
    }
    pathFile.close();
    // Also save several sample paths (for a nicer "spaghetti plot")
    MonteCarloEngine plotEngine(30, numSteps);
    ofstream multiPathFile("multiple_paths.csv");
    multiPathFile << "PathID,Step,Price\n";
    vector<double> dummyZ(numSteps);
    for (int p = 0; p < 30; p++) {
        vector<double> path = plotEngine.generatePath(option, dummyZ, false, false);
        for (size_t i = 0; i < path.size(); i++) {
            multiPathFile << p << "," << i << "," << path[i] << "\n";
        }
    }
    multiPathFile.close();
    cout << "\nAll results saved to results.csv, sample_path.csv, multiple_paths.csv\n";
    cout << "Now run the Python visualization script to see the graphs.\n";
    return 0;
}