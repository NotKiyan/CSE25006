/*
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Link-quality variation fitted to the UVA LoRaWAN trace (Week 5 of this project).
 */

#ifndef CORRELATED_FADING_LOSS_MODEL_H
#define CORRELATED_FADING_LOSS_MODEL_H

#include "ns3/nstime.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/random-variable-stream.h"

#include <map>
#include <utility>

namespace ns3
{
namespace lorawan
{

/**
 * @ingroup lorawan
 *
 * Extra loss per link (pair of nodes) made of three parts, the shape measured
 * on the UVA links:
 *  - a slow Gauss-Markov term (sigma ~1.2 dB, correlation time ~6 days),
 *  - a short Gauss-Markov term (sigma ~1.3 dB, correlation time ~2 h),
 *  - an independent per-packet term (sigma ~5.2 dB).
 * Each link also gets a static log-normal offset (default 0 dB).
 * The Gauss-Markov terms are advanced lazily by the time since the link's
 * last packet: x <- a x + sigma sqrt(1 - a^2) N(0,1), a = exp(-dt / tau).
 */
class CorrelatedFadingLossModel : public PropagationLossModel
{
  public:
    static TypeId GetTypeId();
    CorrelatedFadingLossModel();
    ~CorrelatedFadingLossModel() override;

  private:
    /// State of one link.
    struct Link
    {
        double slow = 0;    ///< slow term [dB]
        double shortT = 0;  ///< short term [dB]
        double offset = 0;  ///< static offset [dB]
        Time last;          ///< time of the last update
        bool init = false;  ///< drawn yet
    };

    double DoCalcRxPower(double txPowerDbm,
                         Ptr<MobilityModel> a,
                         Ptr<MobilityModel> b) const override;
    int64_t DoAssignStreams(int64_t stream) override;

    double m_sigmaSlow;   ///< [dB]
    double m_tauSlowH;    ///< [h]
    double m_sigmaShort;  ///< [dB]
    double m_tauShortH;   ///< [h]
    double m_sigmaFast;   ///< [dB]
    double m_sigmaStatic; ///< [dB]
    double m_extraLossDb; ///< deterministic extra loss for every link [dB] (e.g. a heat cost)
    Ptr<NormalRandomVariable> m_normal; ///< N(0,1)
    mutable std::map<std::pair<const void*, const void*>, Link> m_links; ///< per-link state
};

} // namespace lorawan
} // namespace ns3

#endif
