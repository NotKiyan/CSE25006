#ifndef CLIMATE_LOSS_MODEL_H
#define CLIMATE_LOSS_MODEL_H

#include "climate-trace.h"

#include "ns3/mobility-model.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/traced-callback.h"

#include <map>

namespace ns3
{

/// Deployment topologies from the CASM paper.
enum ClimateTopology
{
    CLIMATE_FREE = 0,
    CLIMATE_COVERED = 1,
    CLIMATE_UNDERGROUND = 2
};

/**
 * Extra path loss (dB) = tempDbPerC*(T-Tref) + humDbPerPct*(H-Href) [+ hot-weather derate].
 * Positive tempDbPerC: hotter = more loss. Negative humDbPerPct: more humid = less loss.
 */
struct ClimateCoeffs
{
    double tempDbPerC{0.0};
    double humDbPerPct{0.0};

    /**
     * Defaults derived from the CASM paper's median-split shifts (Tables V-VII), converted to
     * slopes by ASSUMING a ~10 degC gap between low/high temperature group medians and a ~30 %RH
     * gap for humidity. These gaps are NOT given in the paper; treat as scenario parameters.
     */
    static ClimateCoeffs FromCasm(ClimateTopology t);
};

/**
 * Chainable loss model: base->SetNext(climate). Applies a time-varying climate offset to links
 * touching a registered end device. Topology is assigned per node id.
 */
class ClimateLossModel : public PropagationLossModel
{
  public:
    static TypeId GetTypeId();
    ClimateLossModel();

    void SetTrace(Ptr<ClimateTrace> trace);
    void SetNodeTopology(uint32_t nodeId, ClimateTopology topo);
    void SetDefaultTopology(ClimateTopology topo) { m_defaultTopo = topo; }
    void SetCoefficients(ClimateTopology topo, ClimateCoeffs c);

    /// Extra loss in dB for a given topology at a given simulation time (also returns T, H).
    double GetExtraLossDb(ClimateTopology topo, double simTimeSec, double& T, double& H) const;

    typedef void (*LossTracedCallback)(double temp, double hum, double extraLossDb);

  private:
    double DoCalcRxPower(double txPowerDbm,
                         Ptr<MobilityModel> a,
                         Ptr<MobilityModel> b) const override;
    int64_t DoAssignStreams(int64_t stream) override { return 0; }

    Ptr<ClimateTrace> m_trace;
    std::map<uint32_t, ClimateTopology> m_nodeTopo;
    ClimateCoeffs m_coeffs[3];
    ClimateTopology m_defaultTopo{CLIMATE_FREE};

    double m_tRef{20.0};
    double m_hRef{60.0};
    double m_timeScale{1.0};   //!< trace seconds per simulated second
    double m_startOffset{0.0}; //!< trace seconds at sim t=0
    double m_hotThreshold{35.0};
    double m_hotDerate{0.0}; //!< extra dB per degC above threshold (hardware derating scenario)

    TracedCallback<double, double, double> m_lossTrace;
};

} // namespace ns3

#endif
