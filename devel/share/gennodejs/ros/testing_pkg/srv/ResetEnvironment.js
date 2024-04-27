// Auto-generated. Do not edit!

// (in-package testing_pkg.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class ResetEnvironmentRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.namespace = null;
    }
    else {
      if (initObj.hasOwnProperty('namespace')) {
        this.namespace = initObj.namespace
      }
      else {
        this.namespace = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ResetEnvironmentRequest
    // Serialize message field [namespace]
    bufferOffset = _serializer.string(obj.namespace, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ResetEnvironmentRequest
    let len;
    let data = new ResetEnvironmentRequest(null);
    // Deserialize message field [namespace]
    data.namespace = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.namespace);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'testing_pkg/ResetEnvironmentRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'dc500f131526a67a5c5e87233ccba797';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string namespace
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ResetEnvironmentRequest(null);
    if (msg.namespace !== undefined) {
      resolved.namespace = msg.namespace;
    }
    else {
      resolved.namespace = ''
    }

    return resolved;
    }
};

class ResetEnvironmentResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.success = null;
    }
    else {
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ResetEnvironmentResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ResetEnvironmentResponse
    let len;
    let data = new ResetEnvironmentResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'testing_pkg/ResetEnvironmentResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '358e233cde0c8a8bcfea4ce193f8fc15';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool success
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ResetEnvironmentResponse(null);
    if (msg.success !== undefined) {
      resolved.success = msg.success;
    }
    else {
      resolved.success = false
    }

    return resolved;
    }
};

module.exports = {
  Request: ResetEnvironmentRequest,
  Response: ResetEnvironmentResponse,
  md5sum() { return '4028ee4ec29cddea9b41f419b98a355d'; },
  datatype() { return 'testing_pkg/ResetEnvironment'; }
};
