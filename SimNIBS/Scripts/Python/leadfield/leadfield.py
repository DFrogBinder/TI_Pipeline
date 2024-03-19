from simnibs import sim_struct, run_simnibs

tdcs_lead_field = sim_struct.TDCSLEADFIELD()

# subject folder [m2m_earnie]
tdcs_lead_field.subpath = '/home/cogitatorprime/sandbox/SimNIBS/simnibs4_examples/m2m_sphere'

# output directory
tdcs_lead_field.pathfem = 'leadfield_output_sphere'

# faster solver, but uses ~12GB RAM
tdcs_lead_field.solver_options = 'pardiso'

run_simnibs(tdcs_lead_field)
