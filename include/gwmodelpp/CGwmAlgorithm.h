#ifndef CGWMALGORITHM_H
#define CGWMALGORITHM_H

/**
 * @brief Abstract algorithm class.
 * This class cannot been constructed.
 * It defines some interface commonly used in spatial algorithms:
 * 
 * - Run the algorithm.
 * 
 * Pointer of this type can be put to these classes:
 * 
 * - CGwmGWRBasic
 * - CGwmGWSS
 * 
 */
class CGwmAlgorithm
{
public:

    /**
     * @brief Construct a new CGwmAlgorithm object.
     */
    CGwmAlgorithm();

    /**
     * @brief Destroy the CGwmAlgorithm object.
     */
    virtual ~CGwmAlgorithm();

public:

    /**
     * @brief Run the algorithm.
     */
    virtual void run() = 0;
};

#endif  // CGWMALGORITHM_H