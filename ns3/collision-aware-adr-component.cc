/*
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * See collision-aware-adr-component.h for the papers this follows.
 */

#include "collision-aware-adr-component.h"

#include "class-a-end-device-lorawan-mac.h"
#include "lora-phy.h"

#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/integer.h"
#include "ns3/log.h"
#include "ns3/simulator.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ns3
{
namespace lorawan
{

NS_LOG_COMPONENT_DEFINE("CollisionAwareAdrComponent");
NS_OBJECT_ENSURE_REGISTERED(CollisionAwareAdrComponent);

namespace
{
/// CA-ADR Table II: receiver sensitivity P_Rmin [dBm] for SF7..SF12, BW 125 kHz
/// (noise figure 6 dB plus the SNR floors of Table I).
const double kPrMinDbm[6] = {-124.5, -127.0, -129.5, -132.0, -134.5, -137.0};
} // namespace

TypeId
CollisionAwareAdrComponent::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::CollisionAwareAdrComponent")
            .SetGroupName("lorawan")
            .AddConstructor<CollisionAwareAdrComponent>()
            .SetParent<NetworkControllerComponent>()
            .AddAttribute("Clusters",
                          "K path-loss clusters, each confined to its own channel. "
                          "1 = CA-ADR (Marini et al. 2021); K > 1 = Cluster-ADR "
                          "(Gorrela and Choudhury 2025).",
                          IntegerValue(1),
                          MakeIntegerAccessor(&CollisionAwareAdrComponent::m_clusters),
                          MakeIntegerChecker<int>(1, 16))
            .AddAttribute("HistoryRange",
                          "Received packets averaged for each device's received power "
                          "(CA-ADR: k = 20; Cluster-ADR: 10).",
                          IntegerValue(20),
                          MakeIntegerAccessor(&CollisionAwareAdrComponent::m_historyRange),
                          MakeIntegerChecker<int>(1, 100))
            .AddAttribute("PeriodSeconds",
                          "Reporting period T [s] used in the ALOHA cap "
                          "p_mac = (1 - ToA/T)^(2(n-1)).",
                          DoubleValue(600.0),
                          MakeDoubleAccessor(&CollisionAwareAdrComponent::m_periodS),
                          MakeDoubleChecker<double>(1.0))
            .AddAttribute("RecomputeInterval",
                          "Minimum time between two global allocations (the papers run "
                          "the allocation at the network server without fixing a rate).",
                          TimeValue(Seconds(600)),
                          MakeTimeAccessor(&CollisionAwareAdrComponent::m_recomputeInterval),
                          MakeTimeChecker())
            .AddAttribute("PerClusterCap",
                          "With Clusters > 1, count devices per SF inside each cluster "
                          "(clusters use separate channels) rather than network-wide.",
                          BooleanValue(true),
                          MakeBooleanAccessor(&CollisionAwareAdrComponent::m_perClusterCap),
                          MakeBooleanChecker())
            .AddAttribute("MaxTxPowerDbm",
                          "Transmit power commanded to every device (both papers keep "
                          "the maximum).",
                          DoubleValue(14.0),
                          MakeDoubleAccessor(&CollisionAwareAdrComponent::m_maxTxPowerDbm),
                          MakeDoubleChecker<double>(0.0, 14.0))
            .AddAttribute("Channels",
                          "Default uplink channels available to confine clusters to "
                          "(EU868: 3).",
                          IntegerValue(3),
                          MakeIntegerAccessor(&CollisionAwareAdrComponent::m_nChannels),
                          MakeIntegerChecker<int>(1, 16));
    return tid;
}

CollisionAwareAdrComponent::CollisionAwareAdrComponent()
    : m_lastRun(Seconds(0)),
      m_ran(false)
{
}

CollisionAwareAdrComponent::~CollisionAwareAdrComponent()
{
}

void
CollisionAwareAdrComponent::OnReceivedPacket(Ptr<const Packet> packet,
                                             Ptr<EndDeviceStatus> status,
                                             Ptr<NetworkStatus> networkStatus)
{
}

void
CollisionAwareAdrComponent::OnFailedReply(Ptr<EndDeviceStatus> status,
                                          Ptr<NetworkStatus> networkStatus)
{
}

uint8_t
CollisionAwareAdrComponent::SfToDr(uint8_t sf)
{
    return static_cast<uint8_t>(12 - sf);
}

double
CollisionAwareAdrComponent::TimeOnAirSeconds(uint8_t sf, uint32_t phyBytes) const
{
    LoraTxParameters p;
    p.spreadingFactor = sf;
    p.lowDataRateOptimize = (sf >= 11);
    return LoraPhy::GetTimeOnAir(phyBytes, p).GetSeconds();
}

double
CollisionAwareAdrComponent::AverageRxPowerAtMaxTp(Ptr<EndDeviceStatus> status) const
{
    // Average over the last k received packets of the strongest gateway's power,
    // referred to the maximum transmit power (both papers keep TP at maximum).
    auto list = status->GetReceivedPacketList();
    double sum = 0;
    int n = 0;
    for (auto it = list.rbegin(); it != list.rend() && n < m_historyRange; ++it, ++n)
    {
        double best = -1e9;
        for (const auto& gw : it->second.gwList)
        {
            best = std::max(best, gw.second.rxPower);
        }
        sum += best;
    }
    double tp = status->GetMac()->GetTransmissionPowerDbm();
    return (n > 0 ? sum / n : -1e9) + (m_maxTxPowerDbm - tp);
}

void
CollisionAwareAdrComponent::Allocate(Ptr<NetworkStatus> networkStatus)
{
    struct Dev
    {
        LoraDeviceAddress addr;
        double pr;      // average received power at max TP [dBm]
        uint8_t sfMin;  // lowest SF its power supports
        uint32_t bytes; // PHY payload size of its last packet
        int cluster;
    };

    std::vector<Dev> devs;
    for (const auto& [addr, st] : networkStatus->m_endDeviceStatuses)
    {
        if (static_cast<int>(st->GetReceivedPacketList().size()) < m_historyRange)
        {
            continue; // not enough history yet: neither assigned nor counted
        }
        Dev d;
        d.addr = addr;
        d.pr = AverageRxPowerAtMaxTp(st);
        d.sfMin = 12;
        for (uint8_t sf = 7; sf <= 12; ++sf)
        {
            if (d.pr >= kPrMinDbm[sf - 7])
            {
                d.sfMin = sf;
                break;
            }
        }
        d.bytes = st->GetLastPacketReceivedFromDevice()->GetSize();
        d.cluster = 0;
        devs.push_back(d);
    }
    if (devs.empty())
    {
        return;
    }

    // Cluster-ADR: sort by path loss (strongest first) and split into K equal groups.
    // CA-ADR (K = 1) processes devices in the order the server stores them.
    int k = std::max(1, m_clusters);
    if (k > 1)
    {
        std::stable_sort(devs.begin(), devs.end(), [](const Dev& a, const Dev& b) {
            return a.pr > b.pr;
        });
        size_t per = (devs.size() + k - 1) / k;
        for (size_t i = 0; i < devs.size(); ++i)
        {
            devs[i].cluster = static_cast<int>(i / per);
        }
    }

    int groups = (k > 1 && m_perClusterCap) ? k : 1;
    std::vector<uint8_t> result(devs.size(), 12);
    double pmac = 1.0;
    bool done = false;
    for (int iter = 0; iter < 100 && !done; ++iter)
    {
        // n_max(SF) = floor( 0.5 * log_{1 - ToA/T}(p_mac) ) + 1, per group and SF
        std::vector<std::vector<int>> count(groups, std::vector<int>(6, 0));
        int assigned = 0;
        for (size_t i = 0; i < devs.size(); ++i)
        {
            int g = (groups > 1) ? devs[i].cluster : 0;
            for (uint8_t sf = devs[i].sfMin; sf <= 12; ++sf)
            {
                double x = 1.0 - TimeOnAirSeconds(sf, devs[i].bytes) / m_periodS;
                int nmax = (x <= 0) ? 1
                                    : static_cast<int>(std::floor(0.5 * std::log(pmac) /
                                                                  std::log(x))) +
                                          1;
                if (count[g][sf - 7] < nmax)
                {
                    count[g][sf - 7]++;
                    result[i] = sf;
                    assigned++;
                    break;
                }
            }
        }
        if (assigned == static_cast<int>(devs.size()) || pmac <= 0.01 + 1e-12)
        {
            done = true;
        }
        else
        {
            pmac -= 0.01;
        }
    }

    for (size_t i = 0; i < devs.size(); ++i)
    {
        Assignment a;
        a.sf = result[i];
        a.channel = (k > 1) ? devs[i].cluster % m_nChannels : -1;
        m_assign[devs[i].addr] = a;
    }
    NS_LOG_DEBUG("Allocated " << devs.size() << " devices, p_mac " << pmac);
}

void
CollisionAwareAdrComponent::BeforeSendingReply(Ptr<EndDeviceStatus> status,
                                               Ptr<NetworkStatus> networkStatus)
{
    Ptr<Packet> myPacket = status->GetLastPacketReceivedFromDevice()->Copy();
    LorawanMacHeader mHdr;
    LoraFrameHeader fHdr;
    fHdr.SetAsUplink();
    myPacket->RemoveHeader(mHdr);
    myPacket->RemoveHeader(fHdr);
    if (!fHdr.GetAdr())
    {
        return;
    }

    if (!m_ran || Simulator::Now() - m_lastRun >= m_recomputeInterval)
    {
        Allocate(networkStatus);
        m_lastRun = Simulator::Now();
        m_ran = true;
    }

    auto it = m_assign.find(fHdr.GetAddress());
    if (it == m_assign.end())
    {
        return;
    }
    const Assignment& a = it->second;
    uint8_t curSf = status->GetFirstReceiveWindowSpreadingFactor();
    double curTp = status->GetMac()->GetTransmissionPowerDbm();
    auto sent = m_sentChannel.find(fHdr.GetAddress());
    int curCh = (sent == m_sentChannel.end()) ? -1 : sent->second;

    if (a.sf != curSf || curTp != m_maxTxPowerDbm || a.channel != curCh)
    {
        std::list<int> channels;
        if (a.channel < 0)
        {
            for (int c = 0; c < m_nChannels; ++c)
            {
                channels.push_back(c);
            }
        }
        else
        {
            channels.push_back(a.channel);
        }
        uint8_t tpIndex = static_cast<uint8_t>(7 - m_maxTxPowerDbm / 2);
        status->m_reply.frameHeader.AddLinkAdrReq(SfToDr(a.sf), tpIndex, channels, 1);
        status->m_reply.frameHeader.SetAsDownlink();
        status->m_reply.macHeader.SetFType(LorawanMacHeader::UNCONFIRMED_DATA_DOWN);
        status->m_reply.needsReply = true;
        m_sentChannel[fHdr.GetAddress()] = a.channel;
        NS_LOG_DEBUG("LinkAdrReq to " << fHdr.GetAddress() << ": SF" << unsigned(a.sf)
                                      << ", channel " << a.channel);
    }
}

} // namespace lorawan
} // namespace ns3
