/*
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Collision-aware SF allocation baselines for the Week 5 comparison:
 *
 *   CA-ADR       R. Marini, W. Cerroni, C. Buratti, "A Novel Collision-Aware
 *                Adaptive Data Rate Algorithm for LoRaWAN Networks," IEEE IoT J.
 *                8(4):2670-2680, 2021 -- Algorithm 3.          (Clusters = 1)
 *   Cluster-ADR  A. Gorrela, N. Choudhury, "A Clustering-Based Adaptive Data Rate
 *                Technique for Industrial LoRaWAN-IoT Networks," IEEE IoT J.
 *                12(6):6506-6518, 2025 -- Algorithm 1.         (Clusters = K > 1)
 *
 * Implemented from the papers' algorithms by this project; not the authors' code.
 */

#ifndef COLLISION_AWARE_ADR_COMPONENT_H
#define COLLISION_AWARE_ADR_COMPONENT_H

#include "end-device-status.h"
#include "network-controller-components.h"
#include "network-status.h"

#include "ns3/nstime.h"

#include <map>

namespace ns3
{
namespace lorawan
{

/**
 * @ingroup lorawan
 *
 * Network-level SF assignment that caps how many devices share each SF so that
 * the pure-ALOHA success probability (1 - ToA/T)^(2(n-1)) stays at a target
 * p_mac, lowering p_mac in 0.01 steps until every device has an SF (CA-ADR,
 * Algorithm 3). Each device starts from the lowest SF its average received
 * power supports (P_R >= P_Rmin(SF), Table II of CA-ADR) and is pushed to the
 * next SF while its SF is full. Transmit power stays at the maximum, as in both
 * papers. With Clusters = K > 1 the devices are first sorted by path loss and
 * split into K equal groups, each confined to its own channel (Cluster-ADR,
 * Algorithm 1).
 */
class CollisionAwareAdrComponent : public NetworkControllerComponent
{
  public:
    static TypeId GetTypeId();
    CollisionAwareAdrComponent();
    ~CollisionAwareAdrComponent() override;

    void OnReceivedPacket(Ptr<const Packet> packet,
                          Ptr<EndDeviceStatus> status,
                          Ptr<NetworkStatus> networkStatus) override;
    void BeforeSendingReply(Ptr<EndDeviceStatus> status, Ptr<NetworkStatus> networkStatus) override;
    void OnFailedReply(Ptr<EndDeviceStatus> status, Ptr<NetworkStatus> networkStatus) override;

  private:
    /// One device's assignment.
    struct Assignment
    {
        uint8_t sf = 12;   //!< assigned spreading factor
        int channel = -1;  //!< confined channel index (-1: all default channels)
    };

    void Allocate(Ptr<NetworkStatus> networkStatus);
    double AverageRxPowerAtMaxTp(Ptr<EndDeviceStatus> status) const;
    double TimeOnAirSeconds(uint8_t sf, uint32_t phyBytes) const;
    static uint8_t SfToDr(uint8_t sf);

    int m_clusters;            //!< K; 1 = CA-ADR
    int m_historyRange;        //!< packets averaged for the received power (k)
    double m_periodS;          //!< reporting period T [s] in the ALOHA cap
    Time m_recomputeInterval;  //!< how often the global allocation is recomputed
    bool m_perClusterCap;      //!< count n(SF) per cluster (true) or network-wide
    double m_maxTxPowerDbm;    //!< transmit power kept for every device
    int m_nChannels;           //!< default channels available to confine clusters to

    Time m_lastRun;                                   //!< time of the last allocation
    bool m_ran;                                       //!< an allocation exists
    std::map<LoraDeviceAddress, Assignment> m_assign; //!< current assignment
    std::map<LoraDeviceAddress, int> m_sentChannel;   //!< channel last commanded
};

} // namespace lorawan
} // namespace ns3

#endif
