"""External effect tools (GitHub, Slack, ...).

Tools live here when they have side effects that cross a process boundary.
They are kept behind an abstract interface so agent code can be tested against
fakes without ever touching a network.
"""
