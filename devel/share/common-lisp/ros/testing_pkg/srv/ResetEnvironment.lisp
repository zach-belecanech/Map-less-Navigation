; Auto-generated. Do not edit!


(cl:in-package testing_pkg-srv)


;//! \htmlinclude ResetEnvironment-request.msg.html

(cl:defclass <ResetEnvironment-request> (roslisp-msg-protocol:ros-message)
  ((namespace
    :reader namespace
    :initarg :namespace
    :type cl:string
    :initform ""))
)

(cl:defclass ResetEnvironment-request (<ResetEnvironment-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ResetEnvironment-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ResetEnvironment-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name testing_pkg-srv:<ResetEnvironment-request> is deprecated: use testing_pkg-srv:ResetEnvironment-request instead.")))

(cl:ensure-generic-function 'namespace-val :lambda-list '(m))
(cl:defmethod namespace-val ((m <ResetEnvironment-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader testing_pkg-srv:namespace-val is deprecated.  Use testing_pkg-srv:namespace instead.")
  (namespace m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ResetEnvironment-request>) ostream)
  "Serializes a message object of type '<ResetEnvironment-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'namespace))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'namespace))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ResetEnvironment-request>) istream)
  "Deserializes a message object of type '<ResetEnvironment-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'namespace) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'namespace) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ResetEnvironment-request>)))
  "Returns string type for a service object of type '<ResetEnvironment-request>"
  "testing_pkg/ResetEnvironmentRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ResetEnvironment-request)))
  "Returns string type for a service object of type 'ResetEnvironment-request"
  "testing_pkg/ResetEnvironmentRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ResetEnvironment-request>)))
  "Returns md5sum for a message object of type '<ResetEnvironment-request>"
  "4028ee4ec29cddea9b41f419b98a355d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ResetEnvironment-request)))
  "Returns md5sum for a message object of type 'ResetEnvironment-request"
  "4028ee4ec29cddea9b41f419b98a355d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ResetEnvironment-request>)))
  "Returns full string definition for message of type '<ResetEnvironment-request>"
  (cl:format cl:nil "string namespace~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ResetEnvironment-request)))
  "Returns full string definition for message of type 'ResetEnvironment-request"
  (cl:format cl:nil "string namespace~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ResetEnvironment-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'namespace))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ResetEnvironment-request>))
  "Converts a ROS message object to a list"
  (cl:list 'ResetEnvironment-request
    (cl:cons ':namespace (namespace msg))
))
;//! \htmlinclude ResetEnvironment-response.msg.html

(cl:defclass <ResetEnvironment-response> (roslisp-msg-protocol:ros-message)
  ((success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass ResetEnvironment-response (<ResetEnvironment-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ResetEnvironment-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ResetEnvironment-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name testing_pkg-srv:<ResetEnvironment-response> is deprecated: use testing_pkg-srv:ResetEnvironment-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <ResetEnvironment-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader testing_pkg-srv:success-val is deprecated.  Use testing_pkg-srv:success instead.")
  (success m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ResetEnvironment-response>) ostream)
  "Serializes a message object of type '<ResetEnvironment-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ResetEnvironment-response>) istream)
  "Deserializes a message object of type '<ResetEnvironment-response>"
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ResetEnvironment-response>)))
  "Returns string type for a service object of type '<ResetEnvironment-response>"
  "testing_pkg/ResetEnvironmentResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ResetEnvironment-response)))
  "Returns string type for a service object of type 'ResetEnvironment-response"
  "testing_pkg/ResetEnvironmentResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ResetEnvironment-response>)))
  "Returns md5sum for a message object of type '<ResetEnvironment-response>"
  "4028ee4ec29cddea9b41f419b98a355d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ResetEnvironment-response)))
  "Returns md5sum for a message object of type 'ResetEnvironment-response"
  "4028ee4ec29cddea9b41f419b98a355d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ResetEnvironment-response>)))
  "Returns full string definition for message of type '<ResetEnvironment-response>"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ResetEnvironment-response)))
  "Returns full string definition for message of type 'ResetEnvironment-response"
  (cl:format cl:nil "bool success~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ResetEnvironment-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ResetEnvironment-response>))
  "Converts a ROS message object to a list"
  (cl:list 'ResetEnvironment-response
    (cl:cons ':success (success msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'ResetEnvironment)))
  'ResetEnvironment-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'ResetEnvironment)))
  'ResetEnvironment-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ResetEnvironment)))
  "Returns string type for a service object of type '<ResetEnvironment>"
  "testing_pkg/ResetEnvironment")