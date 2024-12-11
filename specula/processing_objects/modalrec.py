from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputList, InputValue
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat
from specula.data_objects.slopes import Slopes


class Modalrec(BaseProcessingObj):
    '''Modal reconstructor'''

    def __init__(self,
                 nmodes: int=None,
                 recmat: Recmat=None,
                 projmat: Recmat=None,
                 intmat: Intmat=None,
                 polc: bool=False,
                 filtmat = None,
                 identity: bool=False,
                 ncutmodes: int=None,
                 nSlopesToBeDiscarded: int=None,
                 dmNumber: int=0,
                 noProj: bool=False,
                 target_device_idx=None, 
                 precision=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if polc:
            if identity:
                raise ValueError('identity cannot be set with POLC.')
            if ncutmodes is not None:
                raise ValueError('ncutmodes cannot be set with POLC.')
        else:
            if recmat is None:
                if identity:
                    if nmodes is None:
                        raise ValueError('modalrec nmodes key must be set!')
                    recmat = Recmat(self.xp.identity(nmodes),
                                    target_device_idx=target_device_idx, precision=precision)
                elif intmat:
                    if nmodes:
                        nmodesintmat = intmat.size[0]
                        intmat.reduce_size(nmodesintmat - nmodes)
                    if nSlopesToBeDiscarded:
                        intmat.reduce_slopes(nSlopesToBeDiscarded)
                    recmat = Recmat(intmat.intmat,
                                    target_device_idx=target_device_idx, precision=precision)

            if ncutmodes:
                if recmat is not None:
                    recmat.reduce_size(ncutmodes)
                else:
                    print('recmat cannot be reduced because it is null.')


        if recmat is not None:
            if projmat is None and recmat.proj_list and not noProj:
                if dmNumber is not None:
                    if dmNumber <= 0:
                        raise ValueError('dmNumber must be > 0')
                    projmat = Recmat(recmat.proj_list[dmNumber - 1])
                else:
                    raise ValueError('dmNumber (>0) must be defined if projmat_tag is not defined!')

        if filtmat is not None and recmat is not None:
            recmat.recmat = recmat.recmat @ filtmat
            print('recmat updated with filmat!')

        self.recmat = recmat
        self.projmat = projmat
        self.intmat = intmat
        self._polc = polc
        self._layer_modes_list = None
        self._past_step_list = []

        self._modes = BaseValue('output modes from modal reconstructor', target_device_idx=target_device_idx)
        self._pseudo_ol_modes = BaseValue('output POL modes from modal reconstructor', target_device_idx=target_device_idx)
        self._modes_first_step = BaseValue('output (no projection) modes from modal reconstructor', target_device_idx=target_device_idx)

        self.inputs['in_slopes'] = InputValue(type=Slopes, optional=True)
        self.inputs['in_slopes_list'] = InputList(type=Slopes, optional=True)
        self.outputs['out_modes'] = self.out_modes
        self.outputs['out_pseudo_ol_modes'] = self.pseudo_ol_modes
        self.outputs['out_modes_first_step'] = self.modes_first_step
        
        if self._polc:
            self.out_comm = BaseValue('output commands from modal reconstructor', target_device_idx=target_device_idx)
            self.inputs['in_commands'] = InputValue(type=BaseValue, optional=True)
            self.inputs['in_commands_list'] = InputList(type=BaseValue, optional=True)
            self.outputs['out_comm'] = self.out_comm
   

# TODO
    # @in_slopes.setter
    # def in_slopes(self, value: Slopes):
    #     self._slopes = value
    #     if isinstance(value, Slopes):
    #         size = value.size
    #     if self._polc:
    #         self._pseudo_ol_slopes = Slopes(size)

    @property
    def out_modes(self) -> BaseValue:
        return self._modes

    @property
    def out_layer_modes_list(self):
        return self._layer_modes_list

    @property
    def modes_first_step(self) -> BaseValue:
        return self._modes_first_step

    @modes_first_step.setter
    def modes_first_step(self, value):
        self._modes_first_step = value

    @property
    def pseudo_ol_slopes(self):
        return self._pseudo_ol_slopes

    @property
    def pseudo_ol_modes(self) -> BaseValue:
        return self._pseudo_ol_modes

    def trigger_code(self): # , slope_ptr=None
        if self.recmat.recmat is None:
            print("WARNING: modalrec skipping reconstruction because recmat is NULL")
            return

        slopes = self.local_inputs['in_slopes']
        slopes_list = self.local_inputs['in_slopes_list']
        if slopes is None:
            slopes = self.xp.hstack([x.slopes for x in slopes_list])
        else:
            slopes = slopes.slopes

        if self._modes_first_step.generation_time != self.current_time:
            if self._polc:
                commands = self.local_inputs['in_commands']
                commands_list = self.local_inputs['in_commands_list']
                if commands is None:
                    commands = self.xp.hstack([x.commands for x in commands_list])
                else:
                    commands = commands.value
                size = slopes.size
                self._pseudo_ol_slopes = Slopes(size)
                comm_slopes = self.compute_modes(self.intmat, self.xp.array(commands, dtype=self.dtype))
                self._pseudo_ol_slopes.slopes += comm_slopes
                self._pseudo_ol_slopes.generation_time = self.current_time
                m = self.recmat.recmat @ self._pseudo_ol_slopes.slopes
                self._pseudo_ol_modes.value = m
                self._pseudo_ol_modes.generation_time = self.current_time
                if not commands is None:
                    self._modes_first_step.value = m - self.xp.asarray(commands)
                else:
                    self._modes_first_step.value = m
            else:
                m = self.recmat.recmat @ slopes
                self._modes_first_step.value = m
            self._modes_first_step.generation_time = self.current_time
            if self._layer_modes_list is not None:
                for i, idx_list in enumerate(self._layer_idx_list):
                    self._layer_modes_list[i].value = self._modes_first_step.value[idx_list]
                    self._layer_modes_list[i].generation_time = self.current_time

        if self.projmat is None:
            if self._verbose:
                n = len(self._modes_first_step.value)
                print(f"(no projmat) first {min(6, n)} residual values: {self._modes_first_step.value[:min(5, n)]}")
            self._modes.value = self._modes_first_step.value
            self._modes.generation_time = self._modes_first_step.generation_time
        else:
            mp = self.compute_modes(self.projmat, self._modes_first_step.ptr_value)
            if self._verbose:
                print(f"first {min(6, len(mp))} residual values after projection: {mp[:min(5, len(mp))]}")
            self._modes.value = mp
            self._modes.generation_time = self.current_time

        if self._polc and not commands is None:
            self._modes.value -= commands


    def compute_modes(self, matrix, slopes_vector):
        if slopes_vector.shape==():
            m = self.xp.zeros_like(self._pseudo_ol_slopes.slopes)
        else:
            m = self.intmat._intmat @ slopes_vector
        return m

    def setup(self, loop_dt, loop_niters):
        super().setup(loop_dt, loop_niters)

        slopes = self.inputs['in_slopes'].get(self.target_device_idx)
        slopes_list = self.inputs['in_slopes_list'].get(self.target_device_idx)

        if not slopes and not all(slopes_list):
            raise ValueError("Either 'slopes' or 'slopes_list' must be given as an input")
        if not self.recmat:
            raise ValueError("Recmat object not valid")
        if self._polc:
            if not self.intmat:
                raise ValueError("Intmat object not valid")
            #if not self._control_list:
            #    raise ValueError("ControlList object not valid")



