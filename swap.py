'''
StarCluster plugin
Make swap drive for instances on cluster
'''

from starcluster.clustersetup import ClusterSetup
from starcluster.logger import log

class SwapMaker(ClusterSetup):
    def __init__(self, swap_dev):
        self.swap_dev = swap_dev
        log.debug('Making swap on % s' % swap_dev)

    def run(self, nodes, master, user, user_shell, volumes):
        for node in nodes:
            log.info("Unmounting device %s" %self.swap_dev)
            node.ssh.execute("umount -f %s" %self.swap_dev)
            log.info("Making swap on %s" %self.swap_dev)
            node.ssh.execute("mkswap -f %s" %self.swap_dev)
            node.ssh.execute("swapon %s" %self.swap_dev)
