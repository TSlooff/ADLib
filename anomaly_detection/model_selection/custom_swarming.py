import htm.optimization.optimizers as optimizers
from htm.optimization.swarming import ParticleSwarmOptimization, ParticleData, particle_strength, global_strength, velocity_strength
from htm.optimization.parameter_set import ParameterSet

import sys
import os
import random
import pickle
import numpy as np
import math
import re

class ParamFreezingRule:
    """
    Represents rules for freezing parameters based on parameter values.
    """

    def __init__(self, rule) -> None:
        """
        creates paramrule based on given rule represented by either:
        - comma-seperated string of parameters to freeze (independent of parameter values)
        - function which outputs parameters to freeze based on input parameter values (as a dict)

        Note: if key <k> in the parameter dict should be frozen
        the output to freeze it should be: ['<k>']
        """
        if type(rule) == str:
            # rule is simply comma-seperated parameters to freeze
            # so override the fixed output of self.process
            self.fixed_out = self.clean_split(rule)
        else:
            # rule should be the function:
            self.process = rule

    @staticmethod
    def clean_split(s):
        """
        splits given string s over ',' and strips whitespace
        """
        x = s.split(",")
        x = set(map(str.strip, x))
        return x

    def process(self, params):
        # for parameter value - independent freezing the fixed_out is overriden
        # for parameter value - dependent freezing the whole function is overriden
        return self.fixed_out

class CustomParticleData(ParticleData):
    """
    This is an extension of the default particle data, 
    where the change is that you can indicate parameters to be frozen.

    # TODO a way to make parameters depend on each other. e.g. layers = 1 -> layer_1, layer_2 parameters don't matter.
    """
    def __init__(self, initial_parameters, param_rules, param_bindings, swarm=None, ):
        # need to set this first because initialize_velocities is called in 
        # super constructor and overriden here to use param_rules
        super().__init__(initial_parameters, swarm)
        self.param_rules = param_rules
        self.param_bindings = param_bindings

    def initialize_velocities(self, swarm=None):
        # Make a new parameter structure for the velocity data.
        self.velocities = ParameterSet( self.parameters )
        # Iterate through every field in the structure.
        for path in self.parameters.enumerate():
            value = self.parameters.get(path)
            if swarm is not None:
                # Analyse the other particle velocities, so that the new
                # velocity is not too large or too small.
                data = [p.velocities.get(path) for p in swarm if p is not self]
                velocity = np.random.normal(np.mean(data), np.std(data))
            else:
                # New swarm, start with a large random velocity.
                max_percent_change = .10
                uniform = 2 * random.random() - 1
                if isinstance(value, float):
                    velocity = value * uniform * max_percent_change
                elif isinstance(value, int):
                    if abs(value) < 1. / max_percent_change:
                        velocity = uniform # Parameters are rounded, so 50% chance this will mutate.
                    else:
                        velocity = value * uniform * max_percent_change
                else:
                    raise NotImplementedError()
            self.velocities.apply( path, velocity )

    def update_position(self):
        freeze_params = set.union(*[rule.process(self.parameters) for rule in self.param_rules])
        for path in self.parameters.enumerate():
            if path not in freeze_params:
                position = self.parameters.get( path )
                velocity = self.velocities.get( path )
                # self.param_bindings(path, self.parameters) gets the binding function for that path.
                # apply this to position + velocity to ensure a minimum / maximum.
                self.parameters.apply( path, self.param_bindings(path, self.parameters)(position + velocity ))

    def update_velocity(self, global_best):
        freeze_params = set.union(*[rule.process(self.parameters) for rule in self.param_rules])
        for path in self.parameters.enumerate():
            if path not in freeze_params:
                postition     = self.parameters.get( path )
                velocity      = self.velocities.get( path )
                particle_best = self.best.get( path ) if self.best is not None else postition
                global_best_x = global_best.get( path ) if global_best is not None else postition

                # Update velocity.
                particle_bias = (particle_best - postition) * particle_strength * random.random()
                global_bias   = (global_best_x - postition)   * global_strength   * random.random()
                velocity = velocity * velocity_strength + particle_bias + global_bias
                self.velocities.apply( path, velocity )

class CustomParticleSwarmOptimization(ParticleSwarmOptimization):
    """
    Extension of default particle swarm optimization, using the custom particle data class above.
    Allows freezing of parameters. # TODO as well as modeling dependencies between parameters.
    """
    def __init__(self, lab, param_rules, param_bindings, args):
        self.swarm_path    = os.path.join( lab.ae_directory, 'particle_swarm.pickle' )
        if args.clear_scores:
            self.clear_scores()
            sys.exit()
        # Setup the particle swarm.
        self.lab           = lab
        self.swarm         = []
        self.particles     = args.swarming
        self.best          = None
        self.best_score    = None
        assert( self.particles >= args.processes )
        # Try loading an existing particle swarm.
        try:
            self.load()
            if self.particles != len(self.swarm):
                print("Warning: requested number of particles does not match number stored on file.")
        except FileNotFoundError:
            pass
        # Add new particles as necessary.
        while len(self.swarm) < self.particles:
            if self.best is not None:
                new_particle = CustomParticleData( self.best, param_rules, param_bindings)
            else:
                new_particle = CustomParticleData( self.lab.default_parameters, param_rules, param_bindings)
            # Evaluate the default parameters a few times, before branching out
            # to the more experimental stuff.
            if( len(self.swarm) >= 3 ):
                new_particle.update_position()
            self.swarm.append( new_particle )
        # Clear all of the mutex locks before starting.
        for particle in self.swarm:
            particle.lock = False
