/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "correlated-fading-loss-model.h"

#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/mobility-model.h"
#include "ns3/simulator.h"

#include <algorithm>
#include <cmath>

namespace ns3
{
namespace lorawan
{

NS_LOG_COMPONENT_DEFINE("CorrelatedFadingLossModel");
NS_OBJECT_ENSURE_REGISTERED(CorrelatedFadingLossModel);

TypeId
CorrelatedFadingLossModel::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::CorrelatedFadingLossModel")
            .SetParent<PropagationLossModel>()
            .SetGroupName("lorawan")
            .AddConstructor<CorrelatedFadingLossModel>()
            .AddAttribute("SigmaSlowDb", "Slow term standard deviation [dB]", DoubleValue(1.23),
                          MakeDoubleAccessor(&CorrelatedFadingLossModel::m_sigmaSlow),
                          MakeDoubleChecker<double>(0.0))
            .AddAttribute("TauSlowHours", "Slow term correlation time [h]", DoubleValue(142.0),
                          MakeDoubleAccessor(&CorrelatedFadingLossModel::m_tauSlowH),
                          MakeDoubleChecker<double>(0.001))
            .AddAttribute("SigmaShortDb", "Short term standard deviation [dB]", DoubleValue(1.31),
                          MakeDoubleAccessor(&CorrelatedFadingLossModel::m_sigmaShort),
                          MakeDoubleChecker<double>(0.0))
            .AddAttribute("TauShortHours", "Short term correlation time [h]", DoubleValue(2.0),
                          MakeDoubleAccessor(&CorrelatedFadingLossModel::m_tauShortH),
                          MakeDoubleChecker<double>(0.001))
            .AddAttribute("SigmaFastDb", "Independent per-packet term [dB]", DoubleValue(5.19),
                          MakeDoubleAccessor(&CorrelatedFadingLossModel::m_sigmaFast),
                          MakeDoubleChecker<double>(0.0))
            .AddAttribute("SigmaStaticDb", "Static per-link log-normal offset [dB]",
                          DoubleValue(0.0),
                          MakeDoubleAccessor(&CorrelatedFadingLossModel::m_sigmaStatic),
                          MakeDoubleChecker<double>(0.0))
            .AddAttribute("ExtraLossDb", "Deterministic extra loss on every link [dB]",
                          DoubleValue(0.0),
                          MakeDoubleAccessor(&CorrelatedFadingLossModel::m_extraLossDb),
                          MakeDoubleChecker<double>());
    return tid;
}

CorrelatedFadingLossModel::CorrelatedFadingLossModel()
{
    m_normal = CreateObject<NormalRandomVariable>();
    m_normal->SetAttribute("Mean", DoubleValue(0.0));
    m_normal->SetAttribute("Variance", DoubleValue(1.0));
}

CorrelatedFadingLossModel::~CorrelatedFadingLossModel()
{
}

double
CorrelatedFadingLossModel::DoCalcRxPower(double txPowerDbm,
                                         Ptr<MobilityModel> a,
                                         Ptr<MobilityModel> b) const
{
    // One state per unordered pair: uplink and downlink share the slow terms.
    const void* pa = PeekPointer(a);
    const void* pb = PeekPointer(b);
    auto key = std::make_pair(std::min(pa, pb), std::max(pa, pb));
    Link& l = m_links[key];
    Time now = Simulator::Now();
    if (!l.init)
    {
        l.slow = m_sigmaSlow * m_normal->GetValue();
        l.shortT = m_sigmaShort * m_normal->GetValue();
        l.offset = m_sigmaStatic * m_normal->GetValue();
        l.last = now;
        l.init = true;
    }
    else if (now > l.last)
    {
        double dtH = (now - l.last).GetSeconds() / 3600.0;
        double as = std::exp(-dtH / m_tauSlowH);
        double ah = std::exp(-dtH / m_tauShortH);
        l.slow = as * l.slow + m_sigmaSlow * std::sqrt(1 - as * as) * m_normal->GetValue();
        l.shortT = ah * l.shortT + m_sigmaShort * std::sqrt(1 - ah * ah) * m_normal->GetValue();
        l.last = now;
    }
    double fast = m_sigmaFast * m_normal->GetValue();
    // Positive values are gains (better link), as SNR deviations in the trace.
    return txPowerDbm + l.slow + l.shortT + l.offset + fast - m_extraLossDb;
}

int64_t
CorrelatedFadingLossModel::DoAssignStreams(int64_t stream)
{
    m_normal->SetStream(stream);
    return 1;
}

} // namespace lorawan
} // namespace ns3
