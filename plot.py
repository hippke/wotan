
plt.figure(figsize=(4.25, 6.5))

m = 2806.81  #2789.03
w = 0.5
w_window = 0.5

ax1 = plt.subplot(411)
ax1.scatter(t, y, color='black', s=1)
ax1.plot(t, trend1, color='red', linewidth=0.5)
ax1.set_xlim(t.min(), t.max())
#ax1.plot(t, trend2, color='red', linestyle='dashed')
ax1.set_ylabel("Flux")
ax1.set_xlabel("Time (days)")

# Second row
ax2 = plt.subplot(434)
ax2.scatter(t, y, color='black', s=1)
ax2.plot(t, trend1, color='red', linewidth=0.5)
ax2.set_xlim(m-w_window, m+w_window)
ax2.axvspan(m-w/2, m+w/2, alpha=0.2, color='blue')
ax2.set_ylabel("Flux")
ax2.xaxis.set_major_formatter(plt.NullFormatter())

ax3 = plt.subplot(435)
ax3.scatter(t, y, color='black', s=1)
ax3.plot(t, trend1, color='red', linewidth=0.5)
ax3.set_xlim(m-w_window, m+w_window)
ax3.axvspan(m-w/2, m+w/2, alpha=0.2, color='blue')
ax3.yaxis.set_major_formatter(plt.NullFormatter())
ax3.xaxis.set_major_formatter(plt.NullFormatter())

ax4 = plt.subplot(436)

# Third row
ax5 = plt.subplot(437)
ax5.scatter(t, y_filt1, color='black', s=1)
ax5.set_xlim(m-w_window, m+w_window)
#ax4.set_xlabel("Time (days)")
ax5.set_ylabel("Flux")

ax6 = plt.subplot(438)
ax6.scatter(t, y_filt2, color='black', s=1)
ax6.set_xlim(m-w_window, m+w_window)
ax6.yaxis.set_ticks_position('none') 
ax6.yaxis.set_major_formatter(plt.NullFormatter())
#ax5.set_xlabel("Time (days)")

ax5 = plt.subplot(439)

# TLS row
ax7 = plt.subplot(4,3,10)
ax7.set_xlabel("Time (days)")
ax7.set_ylabel("SDE")

#ax6 = plt.subplot(4310)
#ax6.set_xlabel("Time (days)")
#ax6.set_ylabel("SDE")

plt.savefig("1.pdf", bbox_inches='tight')
