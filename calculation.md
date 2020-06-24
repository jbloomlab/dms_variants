\begin{eqnarray}
  \sigma_{s_v}^2
   &=&
   \left(\frac{\partial s_v}{\partial n_v^{\rm{pre}}}\right)^2
   \sigma_{n_v^{\rm{pre}}}^2 +
   \left(\frac{\partial s_v}{\partial n_v^{\rm{post}}}\right)^2
   \sigma_{n_v^{\rm{post}}}^2 \\
   &=&  
   \left(\frac{\partial \log_b B_v}{\partial n_v^{\rm{pre}}}\right)^2
   n_v^{\rm{pre}} +
   \left(\frac{\partial \log_b B_v}{\partial n_v^{\rm{post}}}\right)^2
   n_v^{\rm{post}} \\
   &=&
   \frac{1}{\left(B_v \ln b\right)^2} \left[
   \left(\frac{\partial B_v}{\partial n_v^{\rm{pre}}}\right)^2
   n_v^{\rm{pre}} +
   \left(\frac{\partial B_v}{\partial n_v^{\rm{post}}}\right)^2
   n_v^{\rm{post}}
   \right] \\
   &=&
   \frac{1}{\left(B_v \ln b\right)^2} \left[
   \left(\frac{F N^{\rm{pre}} n_v^{\rm{post}}}{N^{\rm{post}} \left(n_v^{\rm{pre}}\right)^2}\right)^2
   n_v^{\rm{pre}} +
   \left(\frac{F N^{\rm{pre}}}{n_v^{\rm{pre}} N^{\rm{post}}}\right)^2
   n_v^{\rm{post}}
   \right] \\
   &=&
   \left(\frac{F N^{\rm{pre}}}{N^{\rm{post}} B_v \ln b }\right)^2
   \left[\frac{\left(n_v^{\rm{post}}\right)^2}{\left(n_v^{\rm{pre}}\right)^3} + \frac{n_v^{\rm{post}}}{\left(n_v^{\rm{pre}}\right)^2} \right]
\end{eqnarray}