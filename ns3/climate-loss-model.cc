#include "climate-loss-model.h"

#include "ns3/abort.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/simulator.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ClimateLossModel");
NS_OBJECT_ENSURE_REGISTERED(ClimateLossModel);

ClimateCoeffs
ClimateCoeffs::FromCasm(ClimateTopology t)
{
    switch (t)
    {
    case CLIMATE_FREE: // dMed -4.25 dB (temp), +1.5 dB (hum)
        return {0.425, -0.05};
    case CLIMATE_COVERED: // -1.25 / +1.0
        return {0.125, -0.033};
    case CLIMATE_UNDERGROUND: // -2.75 / +1.0
        return {0.275, -0.033};
    }
    return {};
}

TypeId
ClimateLossModel::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::ClimateLossModel")
            .SetParent<PropagationLossModel>()
            .SetGroupName("Climate")
            .AddConstructor<ClimateLossModel>()
            .AddAttribute("TempRef", "Reference temperature (degC) with zero offset",
                          DoubleValue(20.0), MakeDoubleAccessor(&ClimateLossModel::m_tRef),
                          MakeDoubleChecker<double>())
            .AddAttribute("HumRef", "Reference humidity (%RH) with zero offset",
                          DoubleValue(60.0), MakeDoubleAccessor(&ClimateLossModel::m_hRef),
                          MakeDoubleChecker<double>())
            .AddAttribute("TimeScale", "Trace seconds per simulated second (e.g. 86400 = 1 day/s)",
                          DoubleValue(1.0), MakeDoubleAccessor(&ClimateLossModel::m_timeScale),
                          MakeDoubleChecker<double>(0.0))
            .AddAttribute("StartOffset", "Trace time (s) at simulation start",
                          DoubleValue(0.0), MakeDoubleAccessor(&ClimateLossModel::m_startOffset),
                          MakeDoubleChecker<double>())
            .AddAttribute("HotThreshold", "Temperature (degC) above which hardware derating starts",
                          DoubleValue(35.0), MakeDoubleAccessor(&ClimateLossModel::m_hotThreshold),
                          MakeDoubleChecker<double>())
            .AddAttribute("HotDerate", "Extra dB per degC above HotThreshold (scenario parameter)",
                          DoubleValue(0.0), MakeDoubleAccessor(&ClimateLossModel::m_hotDerate),
                          MakeDoubleChecker<double>(0.0))
            .AddTraceSource("ClimateLoss", "temp, hum, extra loss (dB) for each link evaluation",
                            MakeTraceSourceAccessor(&ClimateLossModel::m_lossTrace),
                            "ns3::ClimateLossModel::LossTracedCallback");
    return tid;
}

ClimateLossModel::ClimateLossModel()
{
    m_coeffs[CLIMATE_FREE] = ClimateCoeffs::FromCasm(CLIMATE_FREE);
    m_coeffs[CLIMATE_COVERED] = ClimateCoeffs::FromCasm(CLIMATE_COVERED);
    m_coeffs[CLIMATE_UNDERGROUND] = ClimateCoeffs::FromCasm(CLIMATE_UNDERGROUND);
}

void
ClimateLossModel::SetTrace(Ptr<ClimateTrace> trace)
{
    m_trace = trace;
}

void
ClimateLossModel::SetNodeTopology(uint32_t nodeId, ClimateTopology topo)
{
    m_nodeTopo[nodeId] = topo;
}

void
ClimateLossModel::SetCoefficients(ClimateTopology topo, ClimateCoeffs c)
{
    m_coeffs[topo] = c;
}

double
ClimateLossModel::GetExtraLossDb(ClimateTopology topo, double simTimeSec, double& T, double& H) const
{
    NS_ABORT_MSG_IF(!m_trace, "ClimateLossModel: call SetTrace() first");
    m_trace->Get(simTimeSec * m_timeScale + m_startOffset, T, H);
    const ClimateCoeffs& c = m_coeffs[topo];
    double loss = c.tempDbPerC * (T - m_tRef) + c.humDbPerPct * (H - m_hRef);
    if (T > m_hotThreshold)
    {
        loss += m_hotDerate * (T - m_hotThreshold);
    }
    return loss;
}

double
ClimateLossModel::DoCalcRxPower(double txPowerDbm, Ptr<MobilityModel> a, Ptr<MobilityModel> b) const
{
    // the registered end device (if any) decides the topology; gateways are not registered
    ClimateTopology topo = m_defaultTopo;
    for (auto m : {a, b})
    {
        Ptr<Node> n = m->GetObject<Node>();
        if (n)
        {
            auto it = m_nodeTopo.find(n->GetId());
            if (it != m_nodeTopo.end())
            {
                topo = it->second;
                break;
            }
        }
    }
    double T;
    double H;
    double loss = GetExtraLossDb(topo, Simulator::Now().GetSeconds(), T, H);
    m_lossTrace(T, H, loss);
    return txPowerDbm - loss;
}

} // namespace ns3
